import argparse
import json
import os
import re
import shutil
from datetime import datetime
from glob import glob

general_announcement_content = """To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions).

**[NEW] As of Novemeber 2024, Unsloth now supports vision finetuning!**

You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)"""

installation_content = """%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
"""

installation_kaggle_content = """%%capture
# Kaggle is slow - you'll have to wait 5 minutes for it to install.
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth"""

new_announcement_content_non_vlm = """* We support Llama 3.2 Vision 11B, 90B; Pixtral; Qwen2VL 2B, 7B, 72B; and any Llava variant like Llava NeXT!
* We support 16bit LoRA via `load_in_4bit=False` or 4bit QLoRA. Both are accelerated and use much less memory!
"""

new_announcement_content_vlm = """**We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! You can also select to finetune the attention or the MLP layers!**"""

naming_mapping = {"mistral": ["pixtral"]}


def copy_folder(source_path, new_name, destination_path=None, replace=False):
    if destination_path is None:
        destination_path = os.path.dirname(source_path)

    new_path = os.path.join(destination_path, new_name)

    try:
        if replace and os.path.exists(new_path):
            shutil.rmtree(new_path)
            print(f"Removed existing folder: '{new_path}'")

        shutil.copytree(source_path, new_path)
        print(f"Successfully copied '{source_path}' to '{new_path}'")
    except FileNotFoundError:
        print(f"Error: Source folder '{source_path}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def is_path_contains_any(file_path, words):
    return any(re.search(word, file_path, re.IGNORECASE) for word in words)


def update_notebook_sections(
    notebook_path,
    general_announcement,
    installation_steps,
    installation_steps_kaggle,
    new_announcement_non_vlm,
    new_announcement_vlm,
):
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)

        updated = False
        i = 0
        while i < len(notebook_content["cells"]):
            cell = notebook_content["cells"][i]

            if cell["cell_type"] == "markdown":
                source_str = "".join(cell["source"]).strip()

                if source_str == "# General":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "markdown"
                    ):
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in general_announcement.splitlines()
                        ]
                        updated = True
                        i += 1
                elif source_str == "# News":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "markdown"
                    ):
                        if is_path_contains_any(notebook_path, ["Vision"]):
                            announcement = new_announcement_vlm
                        else:
                            announcement = new_announcement_non_vlm
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in announcement.splitlines()
                        ]
                        updated = True
                        i += 1
                elif source_str == "# Installation":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "code"
                    ):
                        if is_path_contains_any(notebook_path, ["kaggle"]):
                            installation = installation_steps_kaggle
                        else:
                            installation = installation_steps
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in installation.splitlines()
                        ]
                        updated = True
                        i += 1

            i += 1

        # Ensure GPU metadata is set for Colab
        if "metadata" not in notebook_content:
            notebook_content["metadata"] = {}
        if "accelerator" not in notebook_content["metadata"]:
            notebook_content["metadata"]["accelerator"] = "GPU"
            updated = True
        if "colab" not in notebook_content["metadata"]:
            notebook_content["metadata"]["colab"] = {"provenance": []}
            updated = True
        if "kernelspec" not in notebook_content["metadata"]:
            notebook_content["metadata"]["kernelspec"] = {
                "display_name": "Python 3",
                "name": "python3",
            }
            updated = True

        if updated:
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=1)
            print(f"Updated: {notebook_path}")
        else:
            print(f"No sections found to update in: {notebook_path}")

    except FileNotFoundError:
        print(f"Error: Notebook not found at {notebook_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in notebook at {notebook_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {notebook_path}: {e}")


def main():
    notebook_directory = "nb"
    notebook_pattern = "*.ipynb"

    notebook_files = glob(os.path.join(notebook_directory, notebook_pattern))

    if not notebook_files:
        print(
            f"No notebooks found in the directory: {notebook_directory} with pattern: {notebook_pattern}"
        )
        return

    for notebook_file in notebook_files:
        update_notebook_sections(
            notebook_file,
            general_announcement_content,
            installation_content,
            installation_kaggle_content,
            new_announcement_content_non_vlm,
            new_announcement_content_vlm,
        )


def update_readme(
    args,
    readme_path,
    notebooks_dir,
    type_order=None,
    kaggle_accelerator="nvidiaTeslaT4",
):
    if args.to_main_repo:
        base_url_colab = (
            "https://colab.research.google.com/github/unslothai/unsloth/blob/main/nb/"
        )
        base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/unsloth/blob/main/nb/"
    else:
        base_url_colab = (
            "https://colab.research.google.com/github/unslothai/notebooks/blob/main/"
        )
        base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/"

    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))

    list_models = ["Llama", "Phi", "Mistral", "Qwen", "Gemma", "Other notebooks"]
    sections = {}
    for section in list_models:
        sections[section] = {
            "Colab": {
                "header": f"### {section} Notebooks\n",
                "rows": [],
            },
            "Kaggle": {"header": f"### {section} Notebooks\n", "rows": []},
        }

    colab_table_header = "| Model | Type | Colab Link | \n| --- | --- | --- | \n"
    kaggle_table_header = "| Model | Type | Kaggle Link | \n| --- | --- | --- | \n"

    notebook_data = []

    for path in paths:
        notebook_name = os.path.basename(path)
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"])

        section_name = "Other notebooks"

        if is_kaggle:
            link = f"[Open in Kaggle]({base_url_kaggle}{path}"
            # Force to use GPU on start for Kaggle
            if kaggle_accelerator:
                link += f"&accelerator={kaggle_accelerator})"
            else:
                link += ")"
        else:
            link = f"[Open in Colab]({base_url_colab}{path})"
        parts = notebook_name.replace(".ipynb", "").split("-")
        if is_kaggle:
            model = parts[1].replace("_", " ")
        else:
            model = parts[0].replace("_", " ")

        for sect in sections:
            check = [sect.lower()]
            check.extend(naming_mapping.get(sect.lower(), []))
            if is_path_contains_any(path.lower(), check):
                section_name = sect
                break
        type_ = parts[-1].replace("_", " ")
        if is_path_contains_any(path.lower(), ["vision"]):
            type_ = f"**{type_}**"

        notebook_data.append(
            {
                "model": model,
                "type": type_,
                "link": link,
                "section": section_name,
                "path": path,
            }
        )

    if type_order:
        notebook_data.sort(
            key=lambda x: (
                list_models.index(x["section"]),
                type_order.index(x["type"])
                if x["type"] in type_order
                else float("inf"),
            )
        )
    else:
        notebook_data.sort(key=lambda x: (list_models.index(x["section"]), x["type"]))

    for data in notebook_data:
        if is_path_contains_any(data["path"].lower(), ["kaggle"]):
            sections[data["section"]]["Kaggle"]["rows"].append(
                f"| {data['model']} | {data['type']} | {data['link']}\n"
            )
        else:
            sections[data["section"]]["Colab"]["rows"].append(
                f"| {data['model']} | {data['type']} | {data['link']}\n"
            )

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        start_marker = "# 📒 Fine-tuning Notebooks"
        start_index = readme_content.find(start_marker)
        if start_index == -1:
            raise ValueError(f"Start marker '{start_marker}' not found in README.")
        start_index += len(start_marker)

        end_marker = "<!-- End of Notebook Links -->"
        end_index = readme_content.find(end_marker)
        if end_index == -1:
            raise ValueError(f"End marker '{end_marker}' not found in README.")

        content_before = readme_content[:start_index]
        content_after = readme_content[end_index:]

        temp = (
            "(https://github.com/unslothai/unsloth/nb/#-kaggle-notebooks).\n\n"
            if args.to_main_repo
            else "(https://github.com/unslothai/notebooks/#-kaggle-notebooks).\n\n"
        )

        colab_updated_notebooks_links = (
            "Below are our notebooks for Google Colab categorized by model.\n"
            "You can also view our [Kaggle notebooks here]"
            f"{temp}"
        )

        kaggle_updated_notebooks_links = (
            "# 📒 Kaggle Notebooks\n"
            "<details>\n  <summary>   \n"
            "Click for all our Kaggle notebooks categorized by model:\n  "
            "</summary>\n\n"
        )

        for section in list_models:
            colab_updated_notebooks_links += (
                sections[section]["Colab"]["header"] + colab_table_header
            )
            colab_updated_notebooks_links += (
                "".join(sections[section]["Colab"]["rows"]) + "\n"
            )

            kaggle_updated_notebooks_links += (
                sections[section]["Kaggle"]["header"] + kaggle_table_header
            )
            kaggle_updated_notebooks_links += (
                "".join(sections[section]["Kaggle"]["rows"]) + "\n"
            )

        kaggle_updated_notebooks_links += "</details>\n\n"

        timestamp = f"<!-- Last updated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"

        updated_readme_content = (
            content_before
            + "\n"
            + colab_updated_notebooks_links
            + kaggle_updated_notebooks_links
            + timestamp
            + content_after
        )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated_readme_content)

        print(f"Successfully updated {readme_path}")

    except FileNotFoundError:
        print(f"Error: {readme_path} not found.")
    except Exception as e:
        print(f"An error occurred while updating {readme_path}: {e}")


def copy_and_update_notebooks(
    template_dir,
    destination_dir,
    general_announcement,
    installation,
    installation_kaggle,
    new_announcement_non_vlm,
    new_announcement_vlm,
):
    """Copies notebooks from template_dir to destination_dir, updates them, and renames them."""
    template_notebooks = glob(os.path.join(template_dir, "*.ipynb"))

    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    os.makedirs(destination_dir, exist_ok=True)

    for template_notebook_path in template_notebooks:
        notebook_name = os.path.basename(template_notebook_path)

        colab_notebook_name = notebook_name
        destination_notebook_path = os.path.join(destination_dir, colab_notebook_name)

        shutil.copy2(template_notebook_path, destination_notebook_path)
        print(f"Copied '{colab_notebook_name}' to '{destination_dir}'")

        kaggle_notebook_name = "Kaggle-" + notebook_name
        destination_notebook_path = os.path.join(destination_dir, kaggle_notebook_name)

        shutil.copy2(template_notebook_path, destination_notebook_path)
        print(f"Copied '{kaggle_notebook_name}' to '{destination_dir}'")

        update_notebook_sections(
            destination_notebook_path,
            general_announcement,
            installation_kaggle,
            installation_kaggle,
            new_announcement_non_vlm,
            new_announcement_vlm,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to_main_repo",
        action="store_true",
        help="Whether update notebooks and README.md for Unsloth main repository or not. Default is False.",
    )
    args = parser.parse_args()
    copy_and_update_notebooks(
        "original_template",
        "nb",
        general_announcement_content,
        installation_content,
        installation_kaggle_content,
        new_announcement_content_non_vlm,
        new_announcement_content_vlm,
    )
    main()

    notebook_directory = "nb"
    readme_path = "README.md"
    type_order = [
        "Alpaca",
        "Conversational",
        "CPT",
        "DPO",
        "ORPO",
        "Text_Completion",
        "CSV",
        "Inference",
        "Unsloth_Studio",
    ]  # Define your desired order here
    update_readme(args, readme_path, notebook_directory, type_order)
