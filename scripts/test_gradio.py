import gradio as gr
import os

mesh = os.path.abspath("image_output/csie_pingpong_mast3r_altered/merged_tetra_mesh.ply")

def get_url():
    # Return a file component to see what URL Gradio gives it
    return gr.File(mesh)

with gr.Blocks() as demo:
    f = gr.File(value=mesh)
    b = gr.Button("Print URL")
    
if __name__ == "__main__":
    print(f"Mesh path: {mesh}")
    demo.launch(server_port=7861)
