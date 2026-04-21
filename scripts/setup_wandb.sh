#!/bin/bash
# Setup wandb - paste your API key when prompted

echo "=== WandB Setup ==="
echo "If you don't have an account, go to: https://wandb.ai/settings#api"
echo "Copy your API key and paste it below when prompted:"
echo

read -p "Enter your WANDB_API_KEY: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "Error: API key is empty"
    exit 1
fi

export WANDB_API_KEY="$API_KEY"
echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc

echo "WANDB_API_KEY has been added to ~/.bashrc"
echo "You can now run training. If you don't want to use wandb, just:"
echo "  export WANDB_MODE=offline"
echo "Done!"
