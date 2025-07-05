import time
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
    # 'orb_slam_path': '../../ORB_SLAM3',  # Update this path
    'vocabulary_path': '../../ORB_SLAM3/Vocabulary/ORBvoc.txt',  # Relative to current directory
    'config_path': 'camera_conf_for_concurrent.yml',      # Relative to current directory
    'executable_path': '../../ORB_SLAM3/Examples/Monocular/mono_euroc',             # Executable name in current directory
    'working_dir': './slam_data',
    'results_dir': './slam_results',
    'test_images_dir': './test_images' # New: Directory for test images
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
        """Create EuRoC dataset structure for ORB-SLAM3 with image format conversion"""
        session_dir = Path(CONFIG['working_dir']) / session_id
        data_dir = session_dir / 'mav0' / 'cam0' / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        frame_names = []
        
        # Copy/convert frames to EuRoC structure keeping original frame names but ensuring PNG format
        for frame_data in frames:
            if 'image_path' in frame_data:
                # Extract original filename from the path
                original_path = Path(frame_data['image_path'])
                
                # Change extension to .png while keeping the base name
                frame_filename = original_path.stem + '.png'
                frame_path = data_dir / frame_filename
                frame_names.append(frame_filename)
                
                # Convert and save the frame as PNG
                self.convert_image_to_png(frame_data['image_path'], frame_path)
                
            elif 'image_data' in frame_data:
                # If image data is provided, we need a filename
                original_filename = frame_data.get('filename', f"frame_{len(frame_names):06d}")
                
                # Ensure PNG extension
                if not original_filename.lower().endswith('.png'):
                    base_name = Path(original_filename).stem
                    frame_filename = base_name + '.png'
                else:
                    frame_filename = original_filename
                    
                frame_path = data_dir / frame_filename
                frame_names.append(frame_filename)
                
                # Save the image data as PNG
                self.save_frame_as_png(frame_data['image_data'], frame_path)
        
        # Create timestamps file (just filenames without extension, one per line)
        timestamps_path = session_dir / 'mav0' / 'cam0' / 'data.csv'
        with open(timestamps_path, 'w') as f:
            for frame_name in frame_names:
                f.write(f"{Path(frame_name).stem}\n")
        
        return str(session_dir), str(timestamps_path)

    def convert_image_to_png(self, input_path, output_path):
        """Convert image file to PNG format"""
        try:
            # Read the image using OpenCV
            image = cv2.imread(str(input_path))
            
            if image is None:
                raise ValueError(f"Could not read image from {input_path}")
            
            # Save as PNG
            success = cv2.imwrite(str(output_path), image)
            
            if not success:
                raise ValueError(f"Failed to save PNG image to {output_path}")
                
            # print(f"Converted {input_path} to {output_path}")
            
        except Exception as e:
            print(f"Error converting image {input_path} to PNG: {e}")
            raise

    def save_frame_as_png(self, image_data, output_path):
        """Save frame data as PNG file"""
        try:
            if isinstance(image_data, str):
                # Assume base64 encoded image
                import base64
                image_bytes = base64.b64decode(image_data)
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError("Could not decode base64 image data")
                    
            elif isinstance(image_data, np.ndarray):
                # Numpy array
                image = image_data
            else:
                raise ValueError("Unsupported image data format")
            
            # Save as PNG
            success = cv2.imwrite(str(output_path), image)
            
            if not success:
                raise ValueError(f"Failed to save PNG image to {output_path}")
                
            print(f"Saved image data as PNG to {output_path}")
            
        except Exception as e:
            print(f"Error saving image data as PNG: {e}")
            raise
        
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
            # 'KeyFrameTrajectory.txt',
            # 'FrameTrajectory.txt'
        ]
        
        for traj_file in trajectory_files:
            # traj_path = session_dir / traj_file
            if os.path.exists(traj_file):
                results['trajectory'] = self.parse_trajectory_file(traj_file)
                os.remove(traj_file)  # Clean up after parsing
                if os.path.exists("KeyFrameTrajectory.txt"):
                    os.remove("KeyFrameTrajectory.txt")  # Clean up keyframe trajectory if exists
                break
        
        # Extract scale, rotation, translation from trajectory
        if results['trajectory']:
            results = self.extract_pose_data(results)
        
        return results
    
    def parse_trajectory_file(self, file_path):
        """Parse trajectory file from ORB-SLAM3, returning only the last valid pose."""
        last_pose = None
        try:
            with open(file_path, 'r') as f:
                # Read all lines into a list
                lines = f.readlines()
                
                # Iterate from the end of the list to find the last valid line
                for line in reversed(lines):
                    if line.startswith('#') or not line.strip():
                        continue  # Skip comments and empty lines
                    
                    parts = line.strip().split()
                    if len(parts) >= 8:  # timestamp + 7 pose values
                        last_pose = {
                            'timestamp': float(parts[0]),
                            'tx': float(parts[1]),
                            'ty': float(parts[2]),
                            'tz': float(parts[3]),
                            'qx': float(parts[4]),
                            'qy': float(parts[5]),
                            'qz': float(parts[6]),
                            'qw': float(parts[7])
                        }
                        break  # Found the last valid pose, exit the loop
        except Exception as e:
            print(f"Error parsing trajectory file: {e}")
        
        # Return a list containing only the last pose, or an empty list if no valid pose was found
        return [last_pose] if last_pose else []
    
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
    def calculate_global_translation(self, prev_global_translation, prev_local_translation, new_local_translation, session_id):
        """
        Calculate global translation for a new frame based on previous global/local translations
        and the transformation parameters from initial processing.
        
        Args:
            prev_global_translation: Previous frame's global translation [x, y, z]
            prev_local_translation: Previous frame's local translation [x, y, z]
            new_local_translation: New frame's local translation [x, y, z]
            session_id: Session ID to get scale and rotation information
        
        Returns:
            Dict containing the calculated global translation and transformation details
        """
        try:
            # Get the initial processing results for scale and rotation information
            if session_id not in self.processed_results:
                raise ValueError(f"Session {session_id} not found in processed results")
            
            initial_results = self.processed_results[session_id]
            
            # Convert inputs to numpy arrays for easier calculation
            prev_global = np.array(prev_global_translation)
            prev_local = np.array(prev_local_translation)
            new_local = np.array(new_local_translation)
            
            # Calculate local translation delta
            local_delta = new_local - prev_local
            
            # Get scale factor from initial processing
            scale_factor = initial_results.get('scale', 1.0)
            if scale_factor == 0.0:
                scale_factor = 1.0  # Fallback to avoid division by zero
            
            # Apply scale to the local delta
            scaled_delta = local_delta * scale_factor
            
            # Get rotation information from initial processing
            # We'll use the average rotation or the last rotation from initial processing
            rotations = initial_results.get('rotation', [])
            if rotations:
                # Use the last rotation quaternion from initial processing
                last_rotation = rotations[-1]  # [qx, qy, qz, qw]
                
                # Convert quaternion to rotation matrix
                rotation_matrix = self.quaternion_to_rotation_matrix(last_rotation)
                
                # Apply rotation to the scaled delta
                rotated_delta = rotation_matrix @ scaled_delta
            else:
                # No rotation information available, use scaled delta as is
                rotated_delta = scaled_delta
            
            # Calculate new global translation
            new_global_translation = prev_global + rotated_delta
            
            # Calculate distances for validation
            local_distance = np.linalg.norm(local_delta)
            global_distance = np.linalg.norm(rotated_delta)
            
            return {
                'new_global_translation': new_global_translation.tolist(),
                'transformation_details': {
                    'local_delta': local_delta.tolist(),
                    'scaled_delta': scaled_delta.tolist(),
                    'rotated_delta': rotated_delta.tolist(),
                    'scale_factor': scale_factor,
                    'rotation_used': rotations[-1] if rotations else None,
                    'local_distance_moved': local_distance,
                    'global_distance_moved': global_distance
                },
                'input_data': {
                    'prev_global_translation': prev_global_translation,
                    'prev_local_translation': prev_local_translation,
                    'new_local_translation': new_local_translation
                }
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating global translation: {str(e)}")

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.
        """
        qx, qy, qz, qw = quaternion
        
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm == 0:
            return np.eye(3)  # Return identity matrix if quaternion is zero
        
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix
        rotation_matrix = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        return rotation_matrix

# Initialize processor
processor = ORBSLAMProcessor()

@app.route('/server_health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/orb_slam3_health', methods=['GET'])
def orb_slam3_health_check():
    """Health check endpoint for ORB-SLAM3 functionality"""
    test_session_id = f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_images_dir = Path(CONFIG['test_images_dir'])

    if not test_images_dir.exists():
        return jsonify({
            'status': 'error',
            'message': f"Test images directory not found: {test_images_dir}",
            'orb_slam3_status': 'unknown'
        }), 500

    # Collect test images
    test_frames = []
    # Sort the images to maintain a consistent order for processing
    for img_file in sorted(test_images_dir.glob("*.png")):
        test_frames.append({'image_path': str(img_file)})

    if not test_frames:
        return jsonify({
            'status': 'error',
            'message': f"No PNG images found in test images directory: {test_images_dir}",
            'orb_slam3_status': 'unknown'
        }), 500

    # Use a subset of images for a quick health check
    # We only need a few frames to see if ORB-SLAM3 can process them
    frames_for_test = test_frames[:10] 

    try:
        start_time = time.time()
        # Create EuRoC structure for the test frames
        frames_path, timestamps_path = processor.create_euroc_structure(test_session_id, frames_for_test)

        # Run ORB-SLAM3 with the test frames
        success, output = processor.run_orb_slam(frames_path, timestamps_path)

        # Clean up the test session directory
        shutil.rmtree(Path(CONFIG['working_dir']) / test_session_id, ignore_errors=True)

        if success:
            # Parse results to ensure output is as expected, even for a simple run
            results = processor.parse_slam_results(test_session_id)
            if results and results['trajectory']:
                return jsonify({
                    'status': 'healthy',
                    'message': 'ORB-SLAM3 appears to be running correctly.',
                    'orb_slam3_status': 'functional',
                    'processed_test_frames': len(frames_for_test),
                    'timestamp': datetime.now().isoformat(),
                    'local_trajectory': results['trajectory'],
                    'runtime': time.time() - start_time,
                })
            else:
                return jsonify({
                    'status': 'unhealthy',
                    'message': 'ORB-SLAM3 ran, but did not produce a valid trajectory (possible configuration issue).',
                    'orb_slam3_status': 'partially_functional',
                    'details': output,
                    'timestamp': datetime.now().isoformat()
                }), 500
        else:
            return jsonify({
                'status': 'unhealthy',
                'message': 'ORB-SLAM3 execution failed.',
                'orb_slam3_status': 'failed',
                'details': output,
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        # Clean up in case of an exception during testing
        shutil.rmtree(Path(CONFIG['working_dir']) / test_session_id, ignore_errors=True)
        return jsonify({
            'status': 'unhealthy',
            'message': f"An error occurred during ORB-SLAM3 health check: {str(e)}",
            'orb_slam3_status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


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
        results_file = processor.save_results(session_name, results)
        
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
    """Process a specific frame with previous frames provided in request"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        frame_index = data.get('frame_index')
        new_frame = data.get('frame_data')
        session_name = data.get('session_name', 'default')
        
        # NEW: Accept previous frames directly in the request
        previous_frames = data.get('previous_frames', [])
        
        # Optional: Still support legacy session_id lookup as fallback
        session_id = data.get('session_id')
        slice_count = data.get('slice_count', None)
        
        if frame_index is None:
            return jsonify({'error': 'frame_index not provided'}), 400
        
        # Get frames for processing
        frames_to_process = []
        
        # Priority 1: Use provided previous_frames
        if previous_frames:
            frames_to_process = previous_frames.copy()
        
        # Priority 2: Fallback to session_id lookup (legacy support)
        elif session_id and session_id in processor.frames_buffer:
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
            return jsonify({'error': 'No frames available for processing. Provide previous_frames in request or valid session_id'}), 400
        
        # Generate new session ID for this processing
        slice_info = f"_slice{slice_count}" if slice_count is not None else ""
        frames_info = f"_provided{len(previous_frames)}" if previous_frames else ""
        new_session_id = f"{session_name}_frame_{frame_index}{slice_info}{frames_info}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
            'frames_source': 'provided' if previous_frames else 'session_lookup',
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
    """
    Get stored results for a session.
    It first checks for an exact match in memory. If not found, it searches
    the results directory for a filename containing the session_id and
    returns the most recent match.
    """
    # First, check the in-memory cache for an exact match
    if session_id in processor.processed_results:
        return jsonify(processor.processed_results[session_id])

    # If not in memory, search the results directory
    results_dir = Path(CONFIG['results_dir'])
    if not results_dir.exists():
        return jsonify({'error': 'Results directory not found'}), 404

    # Find all files containing the session_id in their name
    matching_files = [f for f in results_dir.iterdir() if session_id in f.name and f.is_file()]

    if not matching_files:
        return jsonify({'error': f'No results found for session key: {session_id}'}), 404

    try:
        # Sort files by modification time (most recent first)
        latest_file = sorted(matching_files, key=os.path.getmtime, reverse=True)[0]

        # Open and return the content of the latest file
        with open(latest_file, 'r') as f:
            results = json.load(f)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'Failed to read or parse results file: {str(e)}'}), 500

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
@app.route('/calculate_vehicle_position', methods=['POST'])
def calculate_vehicle_position():
    """
    Calculate vehicle's global position based on previous global/local translations
    and new local translation using initial processing transformation parameters.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract required parameters
        prev_global_translation = data.get('prev_global_translation')
        prev_local_translation = data.get('prev_local_translation')
        new_local_translation = data.get('new_local_translation')
        session_id = data.get('session_id')
        
        # Validate required parameters
        if prev_global_translation is None:
            return jsonify({'error': 'prev_global_translation not provided'}), 400
        
        if prev_local_translation is None:
            return jsonify({'error': 'prev_local_translation not provided'}), 400
        
        if new_local_translation is None:
            return jsonify({'error': 'new_local_translation not provided'}), 400
        
        if session_id is None:
            return jsonify({'error': 'session_id not provided'}), 400
        
        # Validate translation data format
        for translation_data, name in [
            (prev_global_translation, 'prev_global_translation'),
            (prev_local_translation, 'prev_local_translation'),
            (new_local_translation, 'new_local_translation')
        ]:
            if not isinstance(translation_data, list) or len(translation_data) != 3:
                return jsonify({
                    'error': f'{name} must be a list of 3 numbers [x, y, z]'
                }), 400
            
            try:
                [float(x) for x in translation_data]
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'{name} must contain valid numbers'
                }), 400
        
        # Calculate global translation
        result = processor.calculate_global_translation(
            prev_global_translation,
            prev_local_translation,
            new_local_translation,
            session_id
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'vehicle_global_position': result['new_global_translation'],
            'transformation_details': result['transformation_details'],
            'input_data': result['input_data'],
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False,
            'timestamp': datetime.now().isoformat()
        }), 500
    
    
if __name__ == '__main__':
    print(f"Starting ORB-SLAM3 Flask Server on port {CONFIG['port']}")
    print("Available endpoints:")
    print("  GET  /server_health - Server health check")
    print("  GET  /orb_slam3_health - ORB_SLAM3 health check")
    print("  POST /process_initial - Process first 450 frames")
    print("  POST /process_frame - Process specific frame with history")
    print("  GET  /get_results/<session_id> - Get results for session")
    print("  GET  /list_sessions - List all sessions")
    print("  POST /calculate_vehicle_position - Calculate vehicle's global position")
    app.run(host='0.0.0.0', port=CONFIG['port'], debug=True)