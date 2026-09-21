"""Only the 15 evaluation benchmark task factories are registered."""
from guava.sim.tasks.apple_juice_order import make_apple_juice_order
from guava.sim.tasks.apple_juice_reverse_order import make_apple_juice_reverse_order
from guava.sim.tasks.bin_and_tray_simple import make_bin_and_tray_simple
from guava.sim.tasks.can_in_bin import make_can_in_bin
from guava.sim.tasks.close_drawer import make_close_drawer
from guava.sim.tasks.cube_stack_reverse import make_cube_stack_reverse
from guava.sim.tasks.lemon_in_bin import make_lemon_in_bin
from guava.sim.tasks.pick_up_carrot import make_pick_up_carrot
from guava.sim.tasks.push_basket import make_push_basket
from guava.sim.tasks.push_pot import make_push_pot
from guava.sim.tasks.red_objects_in_basket import make_red_objects_in_basket
from guava.sim.tasks.remove_cube_from_tray import make_remove_cube_from_tray
from guava.sim.tasks.set_table import make_set_table
from guava.sim.tasks.shell_game import make_shell_game
from guava.sim.tasks.tomato_near_potato import make_tomato_near_potato

TASKS = {
    'apple_juice_order': make_apple_juice_order,
    'apple_juice_reverse_order': make_apple_juice_reverse_order,
    'bin_and_tray_simple': make_bin_and_tray_simple,
    'can_in_bin': make_can_in_bin,
    'close_drawer': make_close_drawer,
    'cube_stack_reverse': make_cube_stack_reverse,
    'lemon_in_bin': make_lemon_in_bin,
    'pick_up_carrot': make_pick_up_carrot,
    'push_basket': make_push_basket,
    'push_pot': make_push_pot,
    'red_objects_in_basket': make_red_objects_in_basket,
    'remove_cube_from_tray': make_remove_cube_from_tray,
    'set_table': make_set_table,
    'shell_game': make_shell_game,
    'tomato_near_potato': make_tomato_near_potato,
}

def make_env(cfg):
    from guava.sim.env import RoboEnv
    if cfg.task not in TASKS:
        raise KeyError(f"Unknown task {cfg.task!r}; available: {list(TASKS)}")
    return RoboEnv(TASKS[cfg.task](cfg), cfg)
