import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics
)

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros_like(vectors[0])
    for A, x in zip(matrices, vectors):
        result += A @ x
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(row)) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(col)) for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    if seed is not None:
        np.random.seed(seed)
    M = np.random.normal(mean, std, size=(rows, columns))
    
    return MatrixStatistics(
        matrix=M.tolist(),
        row_means=np.mean(M, axis=1).tolist(),
        column_means=np.mean(M, axis=0).tolist(),
        row_variances=np.var(M, axis=1).tolist(),
        column_variances=np.var(M, axis=0).tolist()
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    M = np.zeros((rows, columns), dtype=int)
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                M[i, j] = first
            else:
                M[i, j] = second
    return M


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    image_height, image_width = data.image_height, data.image_width
    width, height = data.width, data.height
    shape_color, background_color = data.shape_color, data.background_color
    
    img = np.zeros((image_height, image_width, 3), dtype=int)
    img[:, :] = background_color
    start_row = (image_height - height) // 2
    start_col = (image_width - width) // 2
    img[start_row:start_row+height, start_col:start_col+width] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    img = np.zeros((image_height, image_width, 3), dtype=int)
    img[:, :] = background_color
    x0, y0 = image_width / 2, image_height / 2
    for i in range(image_height):
        for j in range(image_width):
            if ((j - x0)**2 / semi_axis_x**2) + ((i - y0)**2 / semi_axis_y**2) <= 1:
                img[i, j] = shape_color
    return img


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.array(data.values), data.window
    n = len(values)
    
    mean = np.mean(values)
    var = np.var(values)
    std = np.std(values)
    
    local_maxima = [i for i in range(1, n-1) if values[i] > values[i-1] and values[i] > values[i+1]]
    local_minima = [i for i in range(1, n-1) if values[i] < values[i-1] and values[i] < values[i+1]]
    
    moving_avg = np.convolve(values, np.ones(window)/window, mode='valid').tolist()
    
    return TimeSeriesStatistics(
        mean=float(mean),
        variance=float(var),
        std=float(std),
        local_maxima_indices=local_maxima, 
        local_minima_indices=local_minima,  
        moving_average=moving_avg
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.array(data.labels), data.class_count
    if class_count is None:
        class_count = np.max(labels) + 1
    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result