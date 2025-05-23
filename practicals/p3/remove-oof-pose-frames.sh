#!/bin/bash

base_path="data/preprocessed_dataset"

if [ ! -d "$base_path" ]; then
    echo "Error: Base path '$base_path' not found."
    exit 1
fi

find "$base_path" -mindepth 1 -maxdepth 1 -type d | sort | while read parent_folder; do
    echo "Checking folder: $parent_folder"

    rgb_path="$parent_folder/rgb"
    pose_path="$parent_folder/pose"
    files_to_delete=()

    declare -A rgb_file_map # Maps frame_id to full_path
    rgb_frame_ids_str=""
    if [ -d "$rgb_path" ]; then
        for file in "$rgb_path"/frame*.png; do
            if [[ -f "$file" && $(basename "$file") =~ frame([0-9]{4}) ]]; then
                frame_id="${BASH_REMATCH[1]}"
                rgb_file_map["$frame_id"]="$file" # Store full path
                rgb_frame_ids_str+="$frame_id"$'\n'
            fi
        done
    else
        echo "Warning: RGB folder not found or is not a directory: $rgb_path"
    fi

    declare -A pose_file_map # Maps frame_id to full_path
    pose_frame_ids_str=""
    if [ -d "$pose_path" ]; then
        for file in "$pose_path"/frame*.png; do
            if [[ -f "$file" && $(basename "$file") =~ frame([0-9]{4}) ]]; then
                frame_id="${BASH_REMATCH[1]}"
                pose_file_map["$frame_id"]="$file" # Store full path
                pose_frame_ids_str+="$frame_id"$'\n'
            fi
        done
    else
        echo "Warning: Pose folder not found or is not a directory: $pose_path"
    fi

    sorted_rgb_ids=$(echo -n "$rgb_frame_ids_str" | sort -u)
    sorted_pose_ids=$(echo -n "$pose_frame_ids_str" | sort -u)

    # Get IDs present in RGB but not in Pose
    rgb_only_ids=$(comm -23 <(echo "$sorted_rgb_ids") <(echo "$sorted_pose_ids"))
    # Get IDs present in Pose but not in RGB
    pose_only_ids=$(comm -13 <(echo "$sorted_rgb_ids") <(echo "$sorted_pose_ids"))

    discrepancy_found=0

    if [ -n "$rgb_only_ids" ]; then
        echo "  Frames in RGB but not in Pose:"
        while IFS= read -r id; do
            # Retrieve the full path from the map
            file_path="${rgb_file_map[$id]}"
            if [ -n "$file_path" ]; then
                echo "    - $id (File: $file_path)"
                files_to_delete+=("$file_path")
            else
                 echo "    - $id (Error: Original file path not found in map for RGB)"
            fi
        done <<< "$rgb_only_ids"
        discrepancy_found=1
    fi

    if [ -n "$pose_only_ids" ]; then
        echo "  Frames in Pose but not in RGB:"
        while IFS= read -r id; do
            # Retrieve the full path from the map
            file_path="${pose_file_map[$id]}"
             if [ -n "$file_path" ]; then
                echo "    - $id (File: $file_path)"
                files_to_delete+=("$file_path")
            else
                echo "    - $id (Error: Original file path not found in map for Pose)"
            fi
        done <<< "$pose_only_ids"
        discrepancy_found=1
    fi

    if [ "$discrepancy_found" -eq 0 ]; then
        if [ -d "$rgb_path" ] || [ -d "$pose_path" ]; then
             echo "  No discrepancies found."
        fi
    elif [ ${#files_to_delete[@]} -gt 0 ]; then
        read -r -p "Do you want to delete these ${#files_to_delete[@]} discrepant files? (y/n) " choice
        case "$choice" in
          y|Y )
            for file_to_delete in "${files_to_delete[@]}"; do
                if [ -f "$file_to_delete" ]; then
                    echo "Deleting file: $file_to_delete"
                    rm -f "$file_to_delete"
                else
                    echo "Warning: File not found, cannot delete: $file_to_delete"
                fi
            done
            ;;
          * )
            echo "Skipping deletion of discrepant files."
            ;;
        esac
    fi
    echo ""
done
