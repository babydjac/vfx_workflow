import os
import glob
import shutil
import zipfile
import torch
from PIL import Image
import gradio as gr
from modules import scripts, script_callbacks, shared, util

extension_root = scripts.basedir()
default_projects_dir = util.truncate_path(os.path.join(extension_root, 'projects'))
shared.options_templates.update(shared.options_section(('saving-paths', 'Paths for saving'), {
    'vfx_workflow_projects_dir': shared.OptionInfo(
        '', 'Projects directory for VFS Workflow', component_args=shared.hide_dirs).info(
        f'if empty use {default_projects_dir}'
    ),
}))


def projects_dir() -> str:
    return shared.opts.vfx_workflow_projects_dir.strip() or default_projects_dir


# Frame Extraction Function
def extract_frames(project_title, video_file):
    """Extract all frames from a video."""
    if not project_title:
        return [], "Error: Project Title is required."

    frame_dir = os.path.join(projects_dir(), project_title, "video_frames")
    os.makedirs(frame_dir, exist_ok=True)

    fps_file = os.path.join(projects_dir(), project_title, "fps.txt")
    video_path = video_file.name

    try:
        # Extract FPS using ffprobe
        fps_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 \"{video_path}\""
        fps = os.popen(fps_cmd).read().strip()

        if not fps:
            raise Exception("Unable to retrieve FPS. Ensure ffprobe is installed.")

        # Save FPS to a file
        with open(fps_file, "w") as f:
            f.write(fps)

        # Extract frames using ffmpeg
        os.system(f"ffmpeg -i \"{video_path}\" -q:v 2 \"{frame_dir}/%05d.png\"")

        # Collect preview frames
        frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])
        if not frame_files:
            raise Exception("No frames were extracted.")

        # Return first 10 frames for preview
        preview_frames = frame_files[:10]
        details = (
            f"Frames extracted to {frame_dir}.\n"
            f"FPS saved to {fps_file}.\n"
            f"Total frames: {len(frame_files)}"
        )
        return preview_frames, details
    except Exception as e:
        return [], f"Error extracting frames: {e}"


# Mask Creation Function
def create_masks(project_title, use_fast_mode=False, use_jit=True):
    """Generate masks for video frames using transparent-background."""
    if not project_title:
        return [], "Error: Project Title is required."

    frame_dir = os.path.join(projects_dir(), project_title, "video_frames")
    mask_dir = os.path.join(projects_dir(), project_title, "video_masks")
    os.makedirs(mask_dir, exist_ok=True)

    try:
        from transparent_background import Remover  # Ensure this package is installed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        remover = Remover(fast=use_fast_mode, jit=use_jit, device=device)

        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if not frame_paths:
            return [], f"No frames found in {frame_dir}. Please check the directory."

        generated_masks = []
        for frame_path in frame_paths:
            base_name = os.path.basename(frame_path)
            image = Image.open(frame_path).convert("RGB")
            mask = remover.process(image, type="map")
            mask = mask.convert("L") if isinstance(mask, Image.Image) else Image.fromarray(mask).convert("L")
            mask_path = os.path.join(mask_dir, base_name)
            mask.save(mask_path)
            generated_masks.append(mask_path)

        return generated_masks[:10], f"Masks saved to {mask_dir}. Total frames processed: {len(frame_paths)}"
    except Exception as e:
        return [], f"Error generating masks: {e}"


# Keyframe Extraction Function
def extract_keyframes(project_title, include_first_last=False, max_keyframes_per_folder=20):
    """Extract keyframes and split into subfolders if needed."""
    if not project_title:
        return [], "Error: Project Title is required."

    frame_dir = os.path.join(projects_dir(), project_title, "video_frames")
    keyframe_dir = os.path.join(projects_dir(), project_title, "keyframes")

    if not os.path.exists(frame_dir):
        return [], "Error: Ensure 'video_frames' directory exists in the project folder."

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if len(frame_paths) < 2:
        return [], "Error: Not enough frames for keyframe extraction."

    os.makedirs(keyframe_dir, exist_ok=True)
    keyframe_indices = []

    # Select keyframe indices
    for i in range(len(frame_paths)):
        if include_first_last and (i == 0 or i == len(frame_paths) - 1):
            keyframe_indices.append(i)
        elif i % 8 == 0:  # Keyframe every 8 frames
            keyframe_indices.append(i)

    total_keyframes = len(keyframe_indices)
    if total_keyframes > max_keyframes_per_folder:
        subfolder_count = (total_keyframes - 1) // max_keyframes_per_folder + 1
        for folder_idx in range(subfolder_count):
            subfolder_path = os.path.join(projects_dir(), project_title, str(folder_idx + 1))
            os.makedirs(subfolder_path, exist_ok=True)

            sub_frame_dir = os.path.join(subfolder_path, "video_frames")
            sub_keyframe_dir = os.path.join(subfolder_path, "keyframes")
            sub_mask_dir = os.path.join(subfolder_path, "video_masks")

            os.makedirs(sub_frame_dir, exist_ok=True)
            os.makedirs(sub_keyframe_dir, exist_ok=True)
            os.makedirs(sub_mask_dir, exist_ok=True)

            start_idx = folder_idx * max_keyframes_per_folder
            end_idx = min(start_idx + max_keyframes_per_folder, total_keyframes)
            for idx in keyframe_indices[start_idx:end_idx]:
                frame_base = os.path.basename(frame_paths[idx])
                shutil.copy(frame_paths[idx], os.path.join(sub_keyframe_dir, frame_base))
    else:
        for idx in keyframe_indices:
            shutil.copy(frame_paths[idx], os.path.join(keyframe_dir, os.path.basename(frame_paths[idx])))

    message = (
        f"Keyframes extracted to {keyframe_dir}.\n"
        f"Subfolders created: {subfolder_count if total_keyframes > max_keyframes_per_folder else 0}.\n"
        f"Total keyframes: {total_keyframes}."
    )
    return frame_paths[:10], message


# Rename img2img_output Files
def rename_img2img_output(project_title):
    """Rename files in the img2img_output folder to remove arbitrary prefixes."""
    if not project_title:
        return "Error: Project Title is required."

    img2img_output_dir = os.path.join(projects_dir(), project_title, "img2img_output")
    if not os.path.exists(img2img_output_dir):
        return f"Error: img2img_output directory not found: {img2img_output_dir}"

    renamed_files = []
    for file in sorted(os.listdir(img2img_output_dir)):
        if "-" in file and file.endswith(".png"):  # Look for files with a prefix separated by "-"
            new_name = file.split("-", 1)[-1]  # Keep everything after the first "-"
            old_path = os.path.join(img2img_output_dir, file)
            new_path = os.path.join(img2img_output_dir, new_name)
            os.rename(old_path, new_path)
            renamed_files.append(new_name)

    if not renamed_files:
        return "No files needed renaming. Ensure the files have a '-' separator in their names."

    return f"Renamed {len(renamed_files)} files in {img2img_output_dir}."


# ZIP File Generation
def generate_zip(project_title):
    """Create a downloadable zip file with video_frames, video_masks, and img2img_output."""
    if not project_title:
        return None, "Error: Project Title is required."

    project_path = os.path.join(projects_dir(), project_title)
    zip_path = os.path.join(projects_dir(), f"{project_title}.zip")

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add folders to the zip file
            for folder_name in ["video_frames", "video_masks", "img2img_output"]:
                folder_path = os.path.join(project_path, folder_name)
                if os.path.exists(folder_path):
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, project_path)
                            zipf.write(file_path, arcname)

        return zip_path, f"ZIP file generated: {zip_path}"
    except Exception as e:
        return None, f"Error creating ZIP file: {e}"


# Generate Img2Img Instructions
def get_batch_instructions(project_title):
    """Provide instructions for inputting paths into the Img2Img Batch Processor."""
    if not project_title:
        return "Error: Project Title is required."

    project_path = os.path.join(projects_dir(), project_title)
    keyframe_dir = os.path.join(project_path, "keyframes")
    output_dir = os.path.join(project_path, "img2img_output")
    mask_dir = os.path.join(project_path, "video_masks")

    if not os.path.exists(keyframe_dir):
        return f"Error: Keyframes directory not found: {keyframe_dir}"
    if not os.path.exists(mask_dir):
        return f"Error: Masks directory not found: {mask_dir}"

    instructions = (
        f"**Img2Img Batch Processing Instructions**:\n\n"
        f"1. **Input Directory**:\n   {keyframe_dir}\n\n"
        f"2. **Output Directory**:\n   {output_dir}\n\n"
        f"3. **Mask Directory**:\n   {mask_dir}\n\n"
        f"Copy these paths into the respective fields in the Img2Img Batch Processor tab."
    )
    return instructions


# UI Tabs
def on_ui_tabs():
    """Define the UI for the VFX Workflow."""
    with gr.Blocks() as vfx_workflow_tab:
        with gr.Tabs() as vfx_tabs:
            # Frame Extraction Tab
            with gr.Tab("Frame Extraction"):
                with gr.Row():
                    project_title = gr.Textbox(label="Project Title")
                    video_file = gr.File(label="Upload Video", file_types=[".mp4", ".mov", ".avi"])
                with gr.Row():
                    extract_frames_btn = gr.Button("Extract Frames")
                with gr.Row():
                    frame_preview = gr.Gallery(label="Preview Extracted Frames", columns=5, show_label=True)
                    extraction_message = gr.Textbox(label="Status", interactive=False)

                extract_frames_btn.click(
                    fn=extract_frames,
                    inputs=[project_title, video_file],
                    outputs=[frame_preview, extraction_message]
                )

            # Mask & Keyframe Extraction Tab
            with gr.Tab("Mask & Keyframe Extraction"):
                with gr.Row():
                    project_title = gr.Textbox(label="Project Title")
                with gr.Row():
                    fast_mode = gr.Checkbox(label="Use Fast Mode")
                    jit_mode = gr.Checkbox(label="Use JIT Compilation", value=True)
                    include_first_last = gr.Checkbox(label="Include First and Last Frames")
                with gr.Row():
                    create_masks_btn = gr.Button("Generate Masks")
                    extract_keyframes_btn = gr.Button("Extract Keyframes")
                with gr.Row():
                    mask_preview = gr.Gallery(label="Preview Masks", columns=5, show_label=True)
                    mask_message = gr.Textbox(label="Mask Status", interactive=False)
                    keyframe_preview = gr.Gallery(label="Preview Keyframes", columns=5, show_label=True)
                    keyframe_message = gr.Textbox(label="Keyframe Status", interactive=False)

                create_masks_btn.click(
                    fn=create_masks,
                    inputs=[project_title, fast_mode, jit_mode],
                    outputs=[mask_preview, mask_message]
                )
                extract_keyframes_btn.click(
                    fn=extract_keyframes,
                    inputs=[project_title, include_first_last],
                    outputs=[keyframe_preview, keyframe_message]
                )

            # Img2Img Instructions Tab
            with gr.Tab("Img2Img Instructions"):
                with gr.Row():
                    project_title = gr.Textbox(label="Project Title")
                with gr.Row():
                    get_instructions_btn = gr.Button("Get Instructions")
                with gr.Row():
                    instructions_message = gr.Textbox(label="Instructions", lines=10, interactive=False)

                get_instructions_btn.click(
                    fn=get_batch_instructions,
                    inputs=[project_title],
                    outputs=[instructions_message]
                )

            # Rename & Generate ZIP Tab
            with gr.Tab("Rename & Generate ZIP"):
                with gr.Row():
                    project_title = gr.Textbox(label="Project Title")
                with gr.Row():
                    rename_files_btn = gr.Button("Rename Img2Img Output Files")
                    generate_zip_btn = gr.Button("Generate ZIP File")
                with gr.Row():
                    rename_message = gr.Textbox(label="Rename Status", interactive=False)
                    zip_file_path = gr.File(label="Download ZIP File")

                rename_files_btn.click(
                    fn=rename_img2img_output,
                    inputs=[project_title],
                    outputs=[rename_message]
                )
                generate_zip_btn.click(
                    fn=generate_zip,
                    inputs=[project_title],
                    outputs=[zip_file_path, rename_message]
                )

    return [(vfx_workflow_tab, "VFX Workflow", "vfx_workflow")]


# Register the UI Tabs
script_callbacks.on_ui_tabs(on_ui_tabs)
