# 1. Use the pre-built geospatial image (includes R, RStudio, and all Spatial/Graphics libs)
FROM rocker/geospatial:4.4.2

# 2. Install Python and Conda dependencies
# We still need these for your SPHARM processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglu1-mesa \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Miniconda (minimal version)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# --- RSTUDIO PROJECT AUTO-LOAD CONFIG ---

# Create the directory where RStudio stores its "last used project" info
RUN mkdir -p /home/rstudio/.local/share/rstudio/projects_settings

# Tell RStudio to load our specific .Rproj file on startup
RUN echo "/project/QSYSpheroidsStudy.Rproj" > /home/rstudio/.local/share/rstudio/projects_settings/last-project-path

# Set global preferences to ensure it opens the project working directory
RUN mkdir -p /home/rstudio/.config/rstudio
RUN echo '{"initial_working_directory": "/project"}' > /home/rstudio/.config/rstudio/rstudio-prefs.json

# Fix permissions so the 'rstudio' user can read/write these configs
RUN chown -R rstudio:rstudio /home/rstudio/.local /home/rstudio/.config

# --- TERMINAL CONFIG ---
RUN echo 'cd /project' >> /home/rstudio/.bashrc

# 4. Set up Project
WORKDIR /project
COPY . /project

# 5. Build Python Env (Using libmamba for speed)
# First, accept the Terms of Service to allow non-interactive installation
# Then, create the environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f py_SPHARM/environment.yml --solver=libmamba

# 6. Restore R packages (Much faster here because system libs are already present)
# Note: rocker/geospatial already has tidyverse and sf, so renv will likely 
# just link to existing high-speed binaries.
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org')"
  
  
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/rstudio/.bashrc
RUN git config --global --add safe.directory /project

#-- How to use this --

# docker build -t spharm .

# Run and map your project folder so changes are saved to our Windows D: drive
# docker run -d -p 8787:8787 -v $(pwd):/project -w /project -e PASSWORD=rstudio spharm

# Open a browser tab at localhost:8787, log in with rstudio/rstudio and ....

# Clean and delete containers. Run on the terminal:
# docker ps -aq | xargs docker stop | xargs docker rm