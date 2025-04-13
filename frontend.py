import streamlit as st
import toml
import subprocess
import os
import re


def read_pyproject_config(file_path="pyproject.toml"):
    """Utility to read the [tool.flwr.app] portion of pyproject.toml."""
    with open(file_path, "r") as f:
        data = toml.load(f)
    flwr_app = data.get("tool", {}).get("flwr", {}).get("app", {})
    return data, flwr_app


def write_pyproject_config(data, new_flwr_app, file_path="pyproject.toml"):
    """Utility to overwrite the [tool.flwr.app] portion of pyproject.toml."""
    data["tool"]["flwr"]["app"] = new_flwr_app
    with open(file_path, "w") as f:
        toml.dump(data, f)


def get_available_models():
    """Define available models and their corresponding server and client apps."""
    return {
        "CNN-BIGRU_multiclass": {
            "serverapp": "Models.final_CNNBIGRU_multiclass.server_app:app",
            "clientapp": "Models.final_CNNBIGRU_multiclass.client_app:app"
        },
        "CNN-BIGRU_binary": {
            "serverapp": "Models.final_CNNBIGRU.server_app:app",
            "clientapp": "Models.final_CNNBIGRU.client_app:app"
        },
        "FMNIST-Baseline": {
            "serverapp": "baselines.fmnist_baseline.server_app:app",
            "clientapp": "baselines.fmnist_baseline.client_app:app"
        },
        "CIFAR-10-Baseline": {
            "serverapp": "baselines.cifar_10_baseline.server_app:app",
            "clientapp": "baselines.cifar_10_baseline.client_app:app"
        }
    }


def clean_log_line(text):
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', text)
    # Remove non-printable and non-ASCII characters
    text = ''.join(c for c in text if c.isprintable() and ord(c) < 128)
    return text


def main():
    # Add a more professional header with project description
    st.set_page_config(
        page_title="FED-NIDS",
        page_icon="🔒",
        layout="wide"
    )

    st.title("🔒 FED-NIDS")
    st.subheader("Decentralised network intrusion detection powered by Flower.ai")

    # Add a visual separator
    st.markdown("<hr style='margin: 15px 0px; height: 1px'>", unsafe_allow_html=True)

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Model Configuration")
        # --- Read existing config ---
        if not os.path.exists("pyproject.toml"):
            st.error("`pyproject.toml` not found in the current directory.")
            return

        data, flwr_app = read_pyproject_config()

        available_models = get_available_models()

        # Determine the currently selected model
        current_serverapp = flwr_app.get("components", {}).get("serverapp", "")
        current_clientapp = flwr_app.get("components", {}).get("clientapp", "")
        selected_model = None
        for model_name, apps in available_models.items():
            if apps["serverapp"] == current_serverapp and apps["clientapp"] == current_clientapp:
                selected_model = model_name
                break
        if selected_model is None and len(available_models) > 0:
            selected_model = list(available_models.keys())[0]  # Default to first model

        # --- Model Selection ---
        model = st.selectbox(
            "Select Model",
            list(available_models.keys()),
            index=list(available_models.keys()).index(selected_model) if selected_model else 0
        )

        # --- Streamlit widgets for Hyperparameters ---
        # Read existing hyperparameters or set defaults
        config = flwr_app.get("config", {})
        default_num_server_rounds = config.get("num-server-rounds", 2)
        default_local_epochs = config.get("local-epochs", 2)
        config.get("fraction-fit", 0.3)
        config.get("min-fit-clients", 3)
        config.get("min-evaluate-clients", 10)
        config.get("min-available-clients", 10)
        config.get("fraction-evaluate", 1.0)
        default_verbose = config.get("verbose", False)
        default_use_wandb = config.get("use-wandb", False)
        default_server_device = config.get("server-device", "cuda:0")
        default_strategy_type = config.get("strategy-type", "reliability_index")

        # Get federation info without allowing customization
        federations_config = data.get("tool", {}).get("flwr", {}).get("federations", {})
        default_federation = data.get("tool", {}).get("flwr", {}).get("federations", {}).get("default", "local-sim-gpu")

        # Remove Federation Setup section and replace with info display
        st.subheader("🌐 Simulation Information")
        federation_info_col, federation_info_col1 = st.columns(2)
        with federation_info_col1:
            st.info(f"Number of selected clients: **3**")
        with federation_info_col:
            default_num_clients = federations_config.get(default_federation, {}).get("options", {}).get(
                "num-supernodes", 10)
            st.info(f"Number of clients: **{default_num_clients}**")

        # Display current resource allocation without allowing modifications
        resource_col1, resource_col2 = st.columns(2)
        with resource_col1:
            default_client_cpus = federations_config.get(default_federation, {}).get("options", {}).get("backend",
                                                                                                        {}).get(
                "client-resources", {}).get("num-cpus", 2)
            st.info(f"CPUs per client: **{default_client_cpus}**")
        with resource_col2:
            default_client_gpus = federations_config.get(default_federation, {}).get("options", {}).get("backend",
                                                                                                        {}).get(
                "client-resources", {}).get("num-gpus", 0.0)
            st.info(f"GPU fraction per client: **{default_client_gpus}**")

        st.subheader("⚙️Training Parameters")
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            num_server_rounds = st.slider("Number of Server Rounds", 1, 50, default_num_server_rounds)
        with param_col2:
            local_epochs = st.slider("Local Epochs", 1, 50, default_local_epochs)

        # Display server device info without customization
        st.info(f"Server device: **{default_server_device}**")

        # Other settings
        st.subheader("📊 Logging Settings")
        log_col1, log_col2 = st.columns(2)
        with log_col1:
            verbose = st.checkbox("Verbose Training Logs", value=bool(default_verbose),
                                  help="Enable detailed client-side training logs")
        with log_col2:
            use_wandb = st.checkbox("Use Weights & Biases", value=default_use_wandb,
                                    help="Enable Weights & Biases integration for experiment tracking")

        # Strategy settings
        st.subheader("🧠 Strategy Settings")
        strategy_options = ["reliability_index", "fedavg"]
        strategy_type = st.selectbox("Strategy Type",
                                     options=strategy_options,
                                     index=strategy_options.index(
                                         default_strategy_type) if default_strategy_type in strategy_options else 0,
                                     help="Reliability Index: Considers client reliability for aggregation\nFedAvg: "
                                          "Standard federated averaging "
                                     )

        # Add alpha parameter selection
        default_alpha = config.get("alpha", 1)
        alpha_options = [0.01, 0.1, 0.5, 1]
        alpha_index = 0  # Default to first option if current value isn't in our options

        # Find if current alpha is in our options
        if default_alpha in alpha_options:
            alpha_index = alpha_options.index(default_alpha)

        alpha = st.radio(
            "Data Heterogeneity Level",
            options=alpha_options,
            index=alpha_index,
            horizontal=True,
            help="Controls the level of data heterogeneity between clients"
        )

        # Create run button - enabled for all strategies
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("🚀 Run Federated Learning", type="primary", use_container_width=True)

        if run_button:
            # Update the [tool.flwr.app.components] based on selected model
            new_components = available_models[model]
            flwr_app["components"] = new_components

            # Get the current complete config to preserve all parameters
            current_config = flwr_app.get("config", {})

            # Update only the specific parameters that are modified in the UI
            # while preserving all other original parameters
            current_config.update({
                "num-server-rounds": num_server_rounds,
                "local-epochs": local_epochs,
                "verbose": verbose,
                "use-wandb": use_wandb,
                "strategy-type": strategy_type,
                "alpha": alpha
            })

            # Set the updated config back
            flwr_app["config"] = current_config

            # Don't modify federation settings - use the defaults from pyproject.toml

            # Write the updated config to pyproject.toml
            write_pyproject_config(data, flwr_app)

            st.success(f"Configuration updated for model: {model}")
            st.info("Starting Flower with the updated configuration...")

            # Run Flower via subprocess
            with st.spinner("Running Flower Federated Learning..."):
                command = ["flwr", "run", "."]

                # Set environment variables to handle encoding issues
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["RAY_DEDUP_LOGS"] = "0"

                # On Windows, force UTF-8 encoding
                if os.name == 'nt':  # Windows
                    env["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"

                try:
                    # Start the subprocess with the environment variables
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        encoding='utf-8',  # Specify encoding
                        errors='replace'  # Replace invalid characters
                    )

                    # Create a better log display
                    st.markdown("### 📈 Training Progress")
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_area = st.empty()

                    log_expander = st.expander("View Training Logs", expanded=True)
                    with log_expander:
                        log_area = st.empty()

                    logs = ""
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        # No need to decode since we specified encoding in Popen
                        clean_line = clean_log_line(line)
                        logs += clean_line + "\n"
                        log_area.code(logs, language="shell")

                        # Update progress based on rounds
                        if "Round" in clean_line:
                            try:
                                current_round = int(re.search(r"Round (\d+)", clean_line).group(1))
                                progress = current_round / num_server_rounds
                                progress_bar.progress(progress)
                                status_area.info(f"Training Progress: Round {current_round}/{num_server_rounds}")
                            except:
                                pass

                    process.stdout.close()
                    return_code = process.wait()

                    if return_code == 0:
                        progress_bar.progress(1.0)
                        st.success("✅ Flower run completed successfully!")
                    else:
                        st.error(f"❌ Flower exited with a non-zero status code: {return_code}")

                except Exception as e:
                    st.error(f"❌ An error occurred while running Flower: {e}")

    with col2:
        st.subheader("🏗️ Model Architecture")
        # Add model architecture descriptions with better styling
        model_descriptions = {
            "CNN-BIGRU_multiclass": """
            ### CNN-BiGRU Multiclass
            
            **Architecture:**
            - CNN layers for feature extraction
            - Bidirectional GRU for temporal dependencies
            - Optimized for multiclass classification
            """,
            "CNN-BIGRU_binary": """
            ### CNN-BiGRU
            
            **Architecture:**
            - CNN layers for spatial features
            - Bidirectional GRU for efficient sequence learning
            - Optimized for binary classification
            """,
            "FMNIST-Baseline": """
            ### FMNIST Baseline
            
            **Architecture:**
            - Simple CNN architecture
            - Optimized for Fashion MNIST dataset
            """,
            "CIFAR-10-Baseline": """
            ### CIFAR-10 Baseline
            
            **Architecture:**
            - Simple CNN architecture
            - Optimized for CIFAR-10 dataset
            """
        }
        st.markdown(model_descriptions.get(model, "Model description not available."))

        # Add a visual representation of the selected model
        st.markdown("#### Model Visualization")
        if model == "CNN-BIGRU_multiclass":
            st.markdown("```\nInput → Conv1D → BiGRU → Dense → Output (Multiclass)\n```")
        elif model == "CNN-BIGRU_binary":
            st.markdown("```\nInput → Conv1D → BiGRU → Dense → Output (Binary)\n```")
        elif model == "FMNIST-Baseline":
            st.markdown("```\nInput → Conv2D → MaxPool → Dense → Output\n```")
        elif model == "CIFAR-10-Baseline":
            st.markdown("```\nInput → Conv2D → MaxPool → Dense → Output\n```")
        else:
            st.markdown("```\nVisualization not available\n```")

    # Show configuration in a more organized way
    st.markdown("<hr style='margin: 20px 0px; height: 1px'>", unsafe_allow_html=True)
    st.subheader("📝 Current Configuration Summary")

    config_tabs = st.tabs(["Model Settings", "Training Parameters", "Federation Settings", "Strategy Parameters"])

    with config_tabs[0]:
        st.json(flwr_app["components"])

    with config_tabs[1]:
        training_params = {k: v for k, v in flwr_app["config"].items() if k in ["num-server-rounds", "local-epochs",
                                                                                "verbose", "use-wandb", "alpha"]}
        st.json(training_params)

    with config_tabs[2]:
        federation_settings = {
            "default": data["tool"]["flwr"]["federations"].get("default", ""),
            "num-supernodes": data["tool"]["flwr"]["federations"].get(default_federation, {}).get("options", {}).get(
                "num-supernodes", ""),
            "client-resources": data["tool"]["flwr"]["federations"].get(default_federation, {}).get("options", {}).get(
                "backend", {}).get("client-resources", {})
        }
        st.json(federation_settings)

    with config_tabs[3]:
        strategy_params = {"strategy-type": flwr_app["config"].get("strategy-type", "reliability_index")}
        st.json(strategy_params)


if __name__ == "__main__":
    main()
