def stamp_brush(grid: list[list[float]], row: int, col: int, radius: int) -> list[list[float]]:
    """
    Returns a new capture grid (grid itself is left untouched) with a radius-cell square brush
    stamped fully on, centered at (row, col), clipped to the grid's bounds. The actual stamping
    algorithm behind both digit_capture.paint_brush_stroke (UCI digits, 32x32 capture grid,
    radius=2) and mnist_capture.paint_brush_stroke (MNIST, 64x64 capture grid, radius=4) - grid
    size and radius are already just parameters here, so the two call sites differ only in
    which values they pass and which grid-size-specific validation they layer on top (see each
    module's own paint_brush_stroke for why the brush was sized the way it was for that
    pipeline).
    """

    grid_size = len(grid)
    assert grid_size >= 1 and all(len(r) == grid_size for r in grid), "grid must be square"
    assert 0 <= row < grid_size and 0 <= col < grid_size, f"(row, col) must be within the grid; got ({row}, {col})"

    new_grid = [list(r) for r in grid]
    for delta_row in range(-radius, radius + 1):
        for delta_col in range(-radius, radius + 1):
            brush_row, brush_col = row + delta_row, col + delta_col
            if 0 <= brush_row < grid_size and 0 <= brush_col < grid_size:
                new_grid[brush_row][brush_col] = 1.0

    return new_grid
