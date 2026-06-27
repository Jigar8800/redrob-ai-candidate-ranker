import gradio as gr
import pandas as pd
import tempfile
import subprocess
import shutil
import os
import sys

def run_ranking(uploaded_file):

    if uploaded_file is None:
        return "❌ Please upload a JSONL file.", None, None

    # uploaded_file is already a filepath in Gradio 6
    input_path = uploaded_file

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "submission.csv")

    try:

        result = subprocess.run(
            [
                sys.executable,
                "rank_candidates.py",
                "--input",
                input_path,
                "--output",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return (
                "❌ Ranking Failed\n\nSTDERR:\n"
                + result.stderr
                + "\n\nSTDOUT:\n"
                + result.stdout,
                None,
                None,
            )

        if not os.path.exists(output_path):
            return "❌ submission.csv was not generated.", None, None

        df = pd.read_csv(output_path)

        return (
            f"✅ Successfully ranked {len(df)} candidates.",
            df,
            output_path,
        )

    except subprocess.TimeoutExpired:
        return "❌ Ranking timed out (300 seconds).", None, None

    except Exception as e:
        return f"❌ {str(e)}", None, None

    #finally:
    #    shutil.rmtree(temp_dir, ignore_errors=True)


with gr.Blocks(title="Drkhack Redrob Ranker") as demo:

    gr.Markdown("# Drkhack Redrob Ranker")
    gr.Markdown("Senior AI Engineer Challenge")

    file_input = gr.File(
        label="Upload sample_candidates.jsonl",
        file_types=[".jsonl"],
        type="filepath",
    )

    run_btn = gr.Button("Run Ranking", variant="primary")

    status = gr.Textbox(label="Status")

    results = gr.Dataframe(label="Results")
    download = gr.File(label="Download submission.csv")

    run_btn.click(
    run_ranking,
    inputs=file_input,
    outputs=[status, results, download],
    )

demo.launch()