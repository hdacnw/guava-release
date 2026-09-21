"""Release inventory: 15 evaluation tasks; collected training data ships separately."""
TRAINING_TASKS = ('apple_juice_order', 'can_in_bin', 'close_drawer', 'push_basket', 'red_objects_in_basket', 'remove_cube_from_tray', 'shell_game')
BENCHMARK_TASKS = ('pick_up_carrot', 'tomato_near_potato', 'lemon_in_bin', 'push_pot', 'cube_stack_reverse', 'apple_juice_reverse_order', 'bin_and_tray_simple', 'set_table', 'can_in_bin', 'apple_juice_order', 'remove_cube_from_tray', 'push_basket', 'close_drawer', 'shell_game', 'red_objects_in_basket')
ALL_TASKS = tuple(sorted(BENCHMARK_TASKS))
