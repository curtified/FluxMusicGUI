import zipfile
import json
import os
import shutil
import uuid

def process_zip_file(zip_file_path: str, base_extraction_dir: str, output_json_name: str = "training_data.json") -> str | None:
    if not os.path.exists(zip_file_path):
        print(f"Error: ZIP file not found at {zip_file_path}")
        return None

    unique_id = os.path.splitext(os.path.basename(zip_file_path))[0] + "_" + str(uuid.uuid4())[:8]
    current_zip_extraction_dir = os.path.join(base_extraction_dir, unique_id)
    
    try:
        os.makedirs(current_zip_extraction_dir, exist_ok=True)
        print(f"Created extraction directory: {current_zip_extraction_dir}")
    except Exception as e:
        print(f"Error creating extraction directory {current_zip_extraction_dir}: {e}")
        return None

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(current_zip_extraction_dir)
        print(f"Successfully extracted ZIP to {current_zip_extraction_dir}")
    except zipfile.BadZipFile:
        print(f"Error: Invalid or corrupted ZIP file at {zip_file_path}")
        return None
    except Exception as e:
        print(f"Error extracting ZIP file {zip_file_path}: {e}")
        return None

    manifest_path = os.path.join(current_zip_extraction_dir, "metadata.jsonl")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest 'metadata.jsonl' not found in {current_zip_extraction_dir}")
        try:
            shutil.rmtree(current_zip_extraction_dir)
            print(f"Cleaned up directory: {current_zip_extraction_dir}")
        except Exception as e_clean:
            print(f"Error cleaning up directory {current_zip_extraction_dir}: {e_clean}")
        return None

    output_json_data = []
    print(f"Processing manifest: {manifest_path}")
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f_manifest:
            for line_num, line in enumerate(f_manifest):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    audio_filename = record.get("audio_filename")
                    prompt = record.get("prompt")

                    if not audio_filename or not prompt:
                        print(f"Warning: Skipping invalid record in manifest (line {line_num + 1}): missing 'audio_filename' or 'prompt'. Record: {line}")
                        continue
                    
                    absolute_audio_path = os.path.abspath(os.path.join(current_zip_extraction_dir, audio_filename))

                    if not os.path.exists(absolute_audio_path):
                        print(f"Warning: Audio file '{audio_filename}' (resolved to {absolute_audio_path}) not found. Skipping (line {line_num + 1}).")
                        continue
                    
                    output_json_data.append({"wav": absolute_audio_path, "label": prompt})
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON in manifest (line {line_num + 1}): {line}")
                    continue
        print(f"Successfully processed {len(output_json_data)} records from manifest.")
    except Exception as e:
        print(f"Error processing manifest file {manifest_path}: {e}")
        return None
        
    if not output_json_data:
        print("No valid records found in manifest or audio files missing. No JSON generated.")
        return None

    final_output_json_path = os.path.join(current_zip_extraction_dir, output_json_name)
    try:
        with open(final_output_json_path, 'w', encoding='utf-8') as f_json_out:
            json.dump(output_json_data, f_json_out, indent=4)
        print(f"Successfully generated training data JSON: {final_output_json_path}")
    except Exception as e:
        print(f"Error writing output JSON to {final_output_json_path}: {e}")
        return None

    return final_output_json_path

if __name__ == '__main__':
    print("Running basic test for data_processor.py...")
    
    test_base_dir = "test_data_processor_output"
    os.makedirs(test_base_dir, exist_ok=True)

    dummy_zip_name = "dummy_audio_package.zip"
    dummy_zip_path = os.path.join(test_base_dir, dummy_zip_name)
    
    try:
        with zipfile.ZipFile(dummy_zip_path, 'w') as zf:
            zf.writestr("audio1.wav", b"dummy audio data 1")
            zf.writestr("track_02.mp3", b"dummy audio data 2")
            zf.writestr("other_files/notes.txt", "some notes")
            metadata_content = (
                '{"audio_filename": "audio1.wav", "prompt": "First dummy audio"}\n'
                '{"audio_filename": "track_02.mp3", "prompt": "Second dummy audio track"}\n'
                '{"audio_filename": "non_existent.wav", "prompt": "This audio does not exist"}\n'
                '{"invalid_json_line"}\n' 
                '{"audio_filename": "audio_no_prompt.wav"}\n'
            )
            zf.writestr("metadata.jsonl", metadata_content)
        print(f"Created dummy ZIP: {dummy_zip_path}")

        generated_json = process_zip_file(dummy_zip_path, test_base_dir, "test_manifest.json")

        if generated_json:
            print(f"Test successful. Generated JSON manifest: {generated_json}")
            with open(generated_json, 'r') as f_test_json:
                print("\nContent of generated JSON:")
                print(f_test_json.read())
        else:
            print("Test failed or no JSON generated.")

    except Exception as e:
        print(f"Error in test harness: {e}")
    finally:
        print(f"Test harness finished. Check contents in: {test_base_dir} (manual cleanup may be needed)")
