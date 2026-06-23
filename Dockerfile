ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-build:uvpipjammy as build

# Switch to Python 3 environment
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install dependencies
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
#RUN install_pinned.py -r /tmp/install/requirements.txt
RUN uv pip compile /tmp/install/requirements.txt \
    -o /tmp/install/requirements.lock && \
    uv pip sync /tmp/install/requirements.lock --strict
# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpdatawriter
#WORKDIR /tmp/install/katsdpdatawriter
#RUN python ./setup.py clean
#RUN pip install --no-deps .
#RUN pip check
RUN cd /tmp/install/katsdpdatawriter && \
    python3 ./setup.py clean   && \
    uv pip install --no-deps . && \
    uv pip check

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-runtime:uvpipjammy
LABEL maintainer="sdpdev+katsdpdatawriter@ska.ac.za"

COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# katcp for vis_writer
EXPOSE 2046
# katcp for flag_writer
EXPOSE 2052
