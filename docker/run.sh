#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Build and push all docker containers

# These containers either function on their own, or are used as base containers for images with
# language runtimes installed.

set -ex

SOLO_DOCKERFILES=('tensorrt')
BASE_DOCKERFILES=('cpu' 'gpu')
LANGUAGE_EXTENSIONS=('r-lang')

for DOCKERFILE_SOLO in "${SOLO_DOCKERFILES[@]}"; do
    docker build -f Dockerfile.${DOCKERFILE_SOLO} . -t "mxnet/${DOCKERFILE_SOLO}:latest"
done

for DOCKERFILE_BASE in "${BASE_DOCKERFILES[@]}"; do
    docker build -f Dockerfile.${DOCKERFILE_BASE} . -t "mxnet/base:latest"
    for DOCKERFILE_LANG in "${LANGUAGE_EXTENSIONS[@]}"; do
        docker build -f Dockerfile.${DOCKERFILE_LANG} . -t "mxnet/${DOCKERFILE_LANG}:latest"
    done
done
