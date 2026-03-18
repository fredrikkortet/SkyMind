import enum
import dataclasses

import jax


class AeroplaneStatus(enum.IntEnum):
    ALIVE = 0
    LOCKED = 1
    CRASHED = 2
    SHOTDOWN = 3
    SUCCESS = 4


def _register_dataclass_pytree(cls):
    """Register a dataclass as a JAX pytree node."""
    fields = dataclasses.fields(cls)

    def flatten(obj):
        children = tuple(getattr(obj, f.name) for f in fields)
        return children, None

    def unflatten(aux_data, children):
        return cls(**{f.name: c for f, c in zip(fields, children)})

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)


@dataclasses.dataclass
class BasePlaneState:
    # Position
    north: jax.typing.ArrayLike = 0.0
    east: jax.typing.ArrayLike = 0.0
    altitude: jax.typing.ArrayLike = 0.0
    # Posture
    roll: jax.typing.ArrayLike = 0.0
    pitch: jax.typing.ArrayLike = 0.0
    yaw: jax.typing.ArrayLike = 0.0
    # velocity
    vel_x: jax.typing.ArrayLike = 0.0
    vel_y: jax.typing.ArrayLike = 0.0
    vel_z: jax.typing.ArrayLike = 0.0
    vt: jax.typing.ArrayLike = 0.0
    status: jax.typing.ArrayLike = AeroplaneStatus.ALIVE.value
    blood: jax.typing.ArrayLike = 100.0
    q0: jax.typing.ArrayLike = 1.0
    q1: jax.typing.ArrayLike = 0.0
    q2: jax.typing.ArrayLike = 0.0
    q3: jax.typing.ArrayLike = 0.0

    @property
    def is_alive(self):
        return self.status == AeroplaneStatus.ALIVE.value

    @property
    def is_locked(self):
        return self.status == AeroplaneStatus.LOCKED.value

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def create(cls, state: jax.Array):
        return cls(
            north=state[0], east=state[1], altitude=state[2],
            roll=state[3], pitch=state[4], yaw=state[5],
            vel_x=state[6], vel_y=state[7], vel_z=state[8],
            vt=state[9],
        )


_register_dataclass_pytree(BasePlaneState)


@dataclasses.dataclass
class BaseControlState:
    throttle: jax.typing.ArrayLike = 0
    elevator: jax.typing.ArrayLike = 0
    aileron: jax.typing.ArrayLike = 0
    rudder: jax.typing.ArrayLike = 0

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def create(cls, action: jax.Array):
        return cls(
            throttle=action[0], elevator=action[1],
            aileron=action[2], rudder=action[3],
        )


_register_dataclass_pytree(BaseControlState)
