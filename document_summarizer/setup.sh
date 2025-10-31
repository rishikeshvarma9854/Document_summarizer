#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit/

# Set up Streamlit config
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml