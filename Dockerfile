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
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org')" \
    && R -e "renv::restore(prompt = FALSE)"


#-- How to use this --

# docker build -t spharm-work .

# Run and map your project folder so changes are saved to your Windows D: drive
# docker run -d \
#   -p 8787:8787 \
#   -v $(pwd):/project \
#   -e PASSWORD=password123 \
#   spharm-work