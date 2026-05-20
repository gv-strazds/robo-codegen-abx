# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base controller for robot articulations (mock version for testing)."""
from abc import ABC, abstractmethod

from isaacsim.core.utils.types import ArticulationAction


class BaseController(ABC):
    """Base controller for robot articulations.
    
    All controllers should inherit from this class to ensure a consistent
    interface for computing and applying articulation actions.

    Args:
        name: Controller name identifier
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Controller name."""
        return self._name

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction:
        """Compute articulation action based on inputs.
        
        A controller should take inputs and return an ArticulationAction
        to be passed to the ArticulationController.

        Returns:
            ArticulationAction with computed joint commands
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset controller state."""
        return
