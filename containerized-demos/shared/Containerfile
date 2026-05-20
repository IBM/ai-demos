ARG BASE_UBI_IMAGE_TAG=9.6
ARG PYTHON_VERSION=3.12

FROM registry.access.redhat.com/ubi9/ubi-minimal:${BASE_UBI_IMAGE_TAG} AS builder

ARG PYTHON_VERSION
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=${VIRTUAL_ENV}/bin:$PATH

# openblas-devel package is needed for numpy package
RUN microdnf install -y dnf && dnf install -y \
    openblas-devel python${PYTHON_VERSION}-devel python${PYTHON_VERSION}-pip \
    && dnf clean all \
    && python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV} \
    && python -m pip install -U pip uv \
    && uv pip install wheel build setuptools 'cmake<4' cython

WORKDIR /app
COPY requirements.txt /app/

RUN  source ${VIRTUAL_ENV}/bin/activate && \
    /opt/venv/bin/pip${PYTHON_VERSION} install --prefer-binary --extra-index-url=https://wheels.developerfirst.ibm.com/ppc64le/linux -r requirements.txt && \
    /opt/venv/bin/pip${PYTHON_VERSION} install --prefer-binary --extra-index-url=https://wheels.developerfirst.ibm.com/ppc64le/linux libprotobuf==4.25.8

FROM registry.access.redhat.com/ubi9/ubi-minimal:${BASE_UBI_IMAGE_TAG}
ARG PYTHON_VERSION

RUN microdnf install -y dnf && dnf install -y \
  python${PYTHON_VERSION} python${PYTHON_VERSION}-devel make && dnf clean all

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/lib64/ /usr/lib64/

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/libprotobuf/lib64/
