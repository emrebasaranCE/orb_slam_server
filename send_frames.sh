#!/bin/bash

API_URL="http://localhost:5000/process_frame"
SESSION_NAME="frame_analysis"
BASE_PATH="/Datasets/Teknofest_2024_butun_oturumlar/TUYZ_Oturum_1/TUYZ_Video_1_mx1vu"

for frame_index in $(seq 1200 4 1300); do
  prev_frames=""
  
  for offset in $(seq 60 -4 4); do
    prev_frame=$((frame_index - offset))
    if [ $prev_frame -ge 0 ]; then
      prev_frame_padded=$(printf "%06d" $prev_frame)
      prev_frames+="
      {
        \"image_path\": \"$BASE_PATH/frame_$prev_frame_padded.jpg\",
        \"filename\": \"frame_$prev_frame_padded.jpg\"
      },"
    fi
  done

  # Remove trailing comma from the last frame
  prev_frames=$(echo "$prev_frames" | sed '$ s/,$//')

  current_frame_padded=$(printf "%06d" $frame_index)
  current_frame_path="$BASE_PATH/frame_$current_frame_padded.jpg"

  echo -e "\nSending frame $frame_index..."

  curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{
      \"frame_index\": $frame_index,
      \"session_name\": \"$SESSION_NAME\",
      \"previous_frames\": [ $prev_frames ],
      \"frame_data\": {
        \"image_path\": \"$current_frame_path\",
        \"filename\": \"frame_$current_frame_padded.jpg\"
      }
    }"

  echo -e "\n✅ Sent frame $frame_index"
done
