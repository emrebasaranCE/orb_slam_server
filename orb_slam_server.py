import os
import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

# Configuration
CONFIG = {
    'port': 5000,
    'orb_slam_path': '/path/to/ORB_SLAM3',  # Update this path
    'vocabulary_path': 'Vocabulary/ORBvoc.txt',  # Relative to current directory
    'config_path': 'Monocular/EuRoC.yaml',      # Relative to current directory
    'executable_path': 'mono_euroc',             # Executable name in current directory
    'working_dir': './slam_data',
    'results_dir': './slam_results'
}

class ORBSLAMProcessor:
    def __init__(self):
        self.frames_buffer = {}  # Store frames for processing
        self.processed_results = {}  # Cache processed results
        self.global_trajectories = {}  # Store global trajectory data
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for processing"""
        for dir_path in [CONFIG['working_dir'], CONFIG['results_dir']]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def create_euroc_structure(self, session_id, frames):
        """Create EuRoC dataset structure for ORB-SLAM3"""
        session_dir = Path(CONFIG['working_dir']) / session_id
        data_dir = session_dir / 'mav0' / 'cam0' / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        frame_names = []
        
        # Copy frames to EuRoC structure keeping original frame names
        for frame_data in frames:
            if 'image_path' in frame_data:
                # Extract original filename from the path
                original_path = Path(frame_data['image_path'])
                frame_filename = original_path.name  # Gets frame_000000.png, frame_000004.png, etc.
                frame_path = data_dir / frame_filename
                frame_names.append(frame_filename)
                
                # Copy the frame
                shutil.copy2(frame_data['image_path'], frame_path)
            elif 'image_data' in frame_data:
                # If image data is provided, we need a filename
                frame_filename = frame_data.get('filename', f"frame_{len(frame_names):06d}.png")
                frame_path = data_dir / frame_filename
                frame_names.append(frame_filename)
                
                # Save the image data
                self.save_frame(frame_data['image_data'], frame_path)
        
        # Create timestamps file (just filenames, one per line)
        timestamps_path = session_dir / 'mav0' / 'cam0' / 'data.csv'
        with open(timestamps_path, 'w') as f:
            for frame_name in frame_names:
                f.write(f"{frame_name}\n")
        
        return str(session_dir), str(timestamps_path)
    
    def save_frame(self, image_data, output_path):
        """Save frame data to file"""
        if isinstance(image_data, str):
            # Assume base64 encoded image
            import base64
            image_bytes = base64.b64decode(image_data)
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
        elif isinstance(image_data, np.ndarray):
            # Numpy array
            cv2.imwrite(str(output_path), image_data)
        else:
            raise ValueError("Unsupported image data format")
    
    def run_orb_slam(self, frames_path, timestamps_path):
        """Execute ORB-SLAM3 processing"""
        try:
            # Stay in current directory - don't change to ORB-SLAM3 directory
            current_dir = os.getcwd()
            
            # Prepare command with full paths
            cmd = [
                CONFIG['executable_path'],           # Executable in current directory
                CONFIG['vocabulary_path'],           # Vocabulary file in current directory
                CONFIG['config_path'],               # Config file in current directory
                frames_path,                         # Frames path (absolute)
                timestamps_path                      # Timestamps path (absolute)
            ]
            
            print(f"Running command from {current_dir}: {' '.join(cmd)}")
            
            # Run ORB-SLAM3 from current directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=current_dir  # Explicitly set working directory to current directory
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, str(e)
    
    def parse_slam_results(self, session_id):
        """Parse ORB-SLAM3 output files and extract results"""
        results = {
            'scale': None,
            'rotation': [],
            'translation': [],
            'keyframes': [],
            'trajectory': []
        }
        
        session_dir = Path(CONFIG['working_dir']) / session_id
        
        # Look for common ORB-SLAM3 output files
        trajectory_files = [
            'CameraTrajectory.txt',
            'KeyFrameTrajectory.txt',
            'FrameTrajectory.txt'
        ]
        
        for traj_file in trajectory_files:
            traj_path = session_dir / traj_file
            if traj_path.exists():
                results['trajectory'] = self.parse_trajectory_file(traj_path)
                break
        
        # Extract scale, rotation, translation from trajectory
        if results['trajectory']:
            results = self.extract_pose_data(results)
        
        return results
    
    def parse_trajectory_file(self, file_path):
        """Parse trajectory file from ORB-SLAM3"""
        trajectory = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 8:  # timestamp + 7 pose values
                        pose = {
                            'timestamp': float(parts[0]),
                            'tx': float(parts[1]),
                            'ty': float(parts[2]),
                            'tz': float(parts[3]),
                            'qx': float(parts[4]),
                            'qy': float(parts[5]),
                            'qz': float(parts[6]),
                            'qw': float(parts[7])
                        }
                        trajectory.append(pose)
        except Exception as e:
            print(f"Error parsing trajectory file: {e}")
        
        return trajectory
    
    def extract_pose_data(self, results):
        """Extract scale, rotation, and translation data from trajectory"""
        if not results['trajectory']:
            return results
        
        trajectory = results['trajectory']
        
        # Calculate scale (distance between consecutive poses)
        scales = []
        for i in range(1, len(trajectory)):
            prev_pose = trajectory[i-1]
            curr_pose = trajectory[i]
            
            dx = curr_pose['tx'] - prev_pose['tx']
            dy = curr_pose['ty'] - prev_pose['ty']
            dz = curr_pose['tz'] - prev_pose['tz']
            
            scale = np.sqrt(dx*dx + dy*dy + dz*dz)
            scales.append(scale)
        
        results['scale'] = np.mean(scales) if scales else 0.0
        
        # Extract rotation (quaternions) and translation
        results['rotation'] = [(p['qx'], p['qy'], p['qz'], p['qw']) for p in trajectory]
        results['translation'] = [(p['tx'], p['ty'], p['tz']) for p in trajectory]
        
        return results
    
    def calculate_global_local_delta(self, global_trajectory, local_trajectory, frame_index):
        """
        Calculate the change between global trajectory position and local trajectory position
        for a specific frame.
        
        Args:
            global_trajectory: Full trajectory from initial 450 frames
            local_trajectory: Local trajectory from recent slice processing
            frame_index: The frame index in the global trajectory
        
        Returns:
            Dict containing the delta information
        """
        if not global_trajectory or not local_trajectory:
            return None
            
        # Get global pose at frame_index
        if frame_index >= len(global_trajectory):
            return None
            
        global_pose = global_trajectory[frame_index]
        
        # Get the last pose from local trajectory (most recent processed frame)
        local_pose = local_trajectory[-1]
        
        # Calculate translation delta (difference between global and local positions)
        translation_delta = {
            'dx': global_pose['tx'] - local_pose['tx'],
            'dy': global_pose['ty'] - local_pose['ty'],
            'dz': global_pose['tz'] - local_pose['tz']
        }
        
        # Calculate distance difference
        distance_delta = np.sqrt(
            translation_delta['dx']**2 + 
            translation_delta['dy']**2 + 
            translation_delta['dz']**2
        )
        
        # Calculate rotation delta (quaternion difference)
        rotation_delta = {
            'dqx': global_pose['qx'] - local_pose['qx'],
            'dqy': global_pose['qy'] - local_pose['qy'],
            'dqz': global_pose['qz'] - local_pose['qz'],
            'dqw': global_pose['qw'] - local_pose['qw']
        }
        
        # Calculate rotation angle difference
        dot_product = (
            global_pose['qx'] * local_pose['qx'] +
            global_pose['qy'] * local_pose['qy'] +
            global_pose['qz'] * local_pose['qz'] +
            global_pose['qw'] * local_pose['qw']
        )
        
        # Clamp dot product to avoid numerical errors
        dot_product = np.clip(abs(dot_product), 0.0, 1.0)
        rotation_angle_delta = 2 * np.arccos(dot_product)
        
        return {
            'translation_delta': translation_delta,
            'rotation_delta': rotation_delta,
            'distance_delta': distance_delta,
            'rotation_angle_delta': rotation_angle_delta,
            'global_pose': {
                'position': [global_pose['tx'], global_pose['ty'], global_pose['tz']],
                'rotation': [global_pose['qx'], global_pose['qy'], global_pose['qz'], global_pose['qw']]
            },
            'local_pose': {
                'position': [local_pose['tx'], local_pose['ty'], local_pose['tz']],
                'rotation': [local_pose['qx'], local_pose['qy'], local_pose['qz'], local_pose['qw']]
            },
            'frame_index': frame_index
        }
    
    def calculate_frame_delta(self, current_results, previous_results):
        """Calculate the change between current and previous frame"""
        if not current_results['trajectory'] or not previous_results['trajectory']:
            return None
        
        # Get the last pose from each trajectory
        current_pose = current_results['trajectory'][-1]
        previous_pose = previous_results['trajectory'][-1]
        
        # Calculate translation delta
        translation_delta = {
            'dx': current_pose['tx'] - previous_pose['tx'],
            'dy': current_pose['ty'] - previous_pose['ty'],
            'dz': current_pose['tz'] - previous_pose['tz']
        }
        
        # Calculate distance moved
        distance_moved = np.sqrt(
            translation_delta['dx']**2 + 
            translation_delta['dy']**2 + 
            translation_delta['dz']**2
        )
        
        # Calculate rotation delta (quaternion difference)
        rotation_delta = {
            'dqx': current_pose['qx'] - previous_pose['qx'],
            'dqy': current_pose['qy'] - previous_pose['qy'],
            'dqz': current_pose['qz'] - previous_pose['qz'],
            'dqw': current_pose['qw'] - previous_pose['qw']
        }
        
        # Calculate rotation magnitude (angle between quaternions)
        dot_product = (
            previous_pose['qx'] * current_pose['qx'] +
            previous_pose['qy'] * current_pose['qy'] +
            previous_pose['qz'] * current_pose['qz'] +
            previous_pose['qw'] * current_pose['qw']
        )
        
        # Clamp dot product to avoid numerical errors
        dot_product = np.clip(abs(dot_product), 0.0, 1.0)
        rotation_angle = 2 * np.arccos(dot_product)
        
        return {
            'translation_delta': translation_delta,
            'rotation_delta': rotation_delta,
            'distance_moved': distance_moved,
            'rotation_angle': rotation_angle,
            'current_pose': {
                'position': [current_pose['tx'], current_pose['ty'], current_pose['tz']],
                'rotation': [current_pose['qx'], current_pose['qy'], current_pose['qz'], current_pose['qw']]
            },
            'previous_pose': {
                'position': [previous_pose['tx'], previous_pose['ty'], previous_pose['tz']],
                'rotation': [previous_pose['qx'], previous_pose['qy'], previous_pose['qz'], previous_pose['qw']]
            }
        }
    
    def save_results(self, session_id, results):
        """Save processing results to file"""
        results_path = Path(CONFIG['results_dir']) / f"{session_id}_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self.make_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return str(results_path)
    
    def make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        else:
            return obj

# Initialize processor
processor = ORBSLAMProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/process_initial', methods=['POST'])
def process_initial_frames():
    """Process first 450 frames and return results"""
    try:
        data = request.get_json()
        
        if not data or 'frames' not in data:
            return jsonify({'error': 'No frames data provided'}), 400
        
        frames = data['frames']
        session_name = data.get('session_name', 'default')  # Get session name from request
        
        if len(frames) >= 450:
            # Limit to first 450 frames
            frames = frames[:450]
        # If less than 450 frames, use all available frames
        else: 
            frames = frames[:len(frames)]

        if len(frames) == 0:
            return jsonify({'error': 'No frames to process'}), 400
        
        # Generate session ID with user-provided name
        session_id = f"{session_name}_initial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create EuRoC structure
        frames_path, timestamps_path = processor.create_euroc_structure(session_id, frames)
        
        # Run ORB-SLAM3
        success, output = processor.run_orb_slam(frames_path, timestamps_path)
        
        if not success:
            return jsonify({
                'error': 'ORB-SLAM3 processing failed',
                'details': output
            }), 500
        
        # Parse results
        results = processor.parse_slam_results(session_id)
        
        # Save results
        results_file = processor.save_results(session_id, results)
        
        # Store frames for potential future processing
        processor.frames_buffer[session_id] = frames
        processor.processed_results[session_id] = results
        
        # Store global trajectory for future comparisons
        processor.global_trajectories[session_id] = results['trajectory']
        
        return jsonify({
            'session_id': session_id,
            'processed_frames': len(frames),
            'scale': results['scale'],
            'rotation': results['rotation'],
            'translation': results['translation'],
            'results_file': results_file,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_single_frame():
    """Process a specific frame with previous frames"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        frame_index = data.get('frame_index')
        session_id = data.get('session_id')
        new_frame = data.get('frame_data')
        slice_count = data.get('slice_count', None)  # How many frames to slice backwards
        session_name = data.get('session_name', 'default')  # Get session name from request
        
        if frame_index is None:
            return jsonify({'error': 'frame_index not provided'}), 400
        
        # Get frames for processing
        frames_to_process = []
        
        if session_id and session_id in processor.frames_buffer:
            # Use existing frames and slice backwards from frame_index
            existing_frames = processor.frames_buffer[session_id]
            
            if slice_count is not None:
                # Slice backwards by slice_count from frame_index
                start_index = max(0, frame_index - slice_count + 1)
                frames_to_process = existing_frames[start_index:frame_index + 1]
            else:
                # Use all frames up to frame_index
                frames_to_process = existing_frames[:frame_index + 1]
        
        # Add new frame if provided
        if new_frame:
            frames_to_process.append(new_frame)
        
        if not frames_to_process:
            return jsonify({'error': 'No frames available for processing'}), 400
        
        # Generate new session ID for this processing with user-provided name
        slice_info = f"_slice{slice_count}" if slice_count is not None else ""
        new_session_id = f"{session_name}_frame_{frame_index}{slice_info}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create EuRoC structure
        frames_path, timestamps_path = processor.create_euroc_structure(new_session_id, frames_to_process)
        
        # Run ORB-SLAM3
        success, output = processor.run_orb_slam(frames_path, timestamps_path)
        
        if not success:
            return jsonify({
                'error': 'ORB-SLAM3 processing failed',
                'details': output
            }), 500
        
        # Parse results
        results = processor.parse_slam_results(new_session_id)
        
        # Calculate frame delta if we have previous results
        frame_delta = None
        if session_id and session_id in processor.processed_results:
            previous_results = processor.processed_results[session_id]
            frame_delta = processor.calculate_frame_delta(results, previous_results)
        
        # Calculate global-local delta if we have global trajectory
        global_local_delta = None
        if session_id and session_id in processor.global_trajectories:
            global_trajectory = processor.global_trajectories[session_id]
            local_trajectory = results['trajectory']
            global_local_delta = processor.calculate_global_local_delta(
                global_trajectory, local_trajectory, frame_index
            )
        
        # Save results
        results_file = processor.save_results(new_session_id, results)
        
        # Prepare response
        response_data = {
            'session_id': new_session_id,
            'frame_index': frame_index,
            'slice_count': slice_count,
            'processed_frames': len(frames_to_process),
            'scale': results['scale'],
            'rotation': results['rotation'],
            'translation': results['translation'],
            'results_file': results_file,
            'success': True
        }
        
        # Add frame delta information if available
        if frame_delta:
            response_data['frame_delta'] = frame_delta
        
        # Add global-local delta information if available
        if global_local_delta:
            response_data['global_local_delta'] = global_local_delta
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_results/<session_id>', methods=['GET'])
def get_results(session_id):
    """Get stored results for a session"""
    if session_id in processor.processed_results:
        return jsonify(processor.processed_results[session_id])
    else:
        results_file = Path(CONFIG['results_dir']) / f"{session_id}_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({'error': 'Results not found'}), 404

@app.route('/list_sessions', methods=['GET'])
def list_sessions():
    """List all available sessions"""
    sessions = list(processor.processed_results.keys())
    
    # Also check for saved result files
    results_dir = Path(CONFIG['results_dir'])
    if results_dir.exists():
        for results_file in results_dir.glob("*_results.json"):
            session_id = results_file.stem.replace('_results', '')
            if session_id not in sessions:
                sessions.append(session_id)
    
    return jsonify({'sessions': sessions})

if __name__ == '__main__':
    print(f"Starting ORB-SLAM3 Flask Server on port {CONFIG['port']}")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /process_initial - Process first 450 frames")
    print("  POST /process_frame - Process specific frame with history")
    print("  GET  /get_results/<session_id> - Get results for session")
    print("  GET  /list_sessions - List all sessions")
    
    app.run(host='0.0.0.0', port=CONFIG['port'], debug=True)