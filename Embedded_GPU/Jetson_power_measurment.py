# %%
# Cell 2: PyTorchESN Class Definition
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

class PyTorchESN(nn.Module):
    def __init__(self,
                 input_dim,
                 reservoir_dim,
                 output_dim,
                 spectral_radius=0.9,
                 W_res_density=0.1,
                 input_scaling=1.0,
                 bias_scaling=0.2,
                 leak_rate=1.0,
                 activation_func=torch.tanh,
                 Win_data=None,
                 W_res_data=None,
                 bias_res_data=None,
                 W_out_data=None,
                 bias_out_data=None,
                 readout_uses_input=False):

        super(PyTorchESN, self).__init__()

        # ---- Store Hyperparameters and Configuration ----
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.W_res_density = W_res_density
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.leak_rate = leak_rate
        self.activation_func = activation_func
        self.readout_uses_input = readout_uses_input

        # This method handles all weight initializations
        self.load_weights_from_data(Win_data, W_res_data, bias_res_data, W_out_data, bias_out_data)

        self.current_reservoir_state = None
        self.is_initial_state = True

    def _initialize_W_res(self):
        W_res_ = torch.randn(self.reservoir_dim, self.reservoir_dim)
        num_zero_elements = int((1 - self.W_res_density) * self.reservoir_dim**2)
        if 0 < num_zero_elements < self.reservoir_dim**2:
            zero_indices = torch.randperm(self.reservoir_dim**2)[:num_zero_elements]
            W_res_.view(-1)[zero_indices] = 0
        
        if self.reservoir_dim > 0:
            try:
                eigenvalues = torch.linalg.eigvals(W_res_)
                current_spectral_radius = torch.max(torch.abs(eigenvalues))
                if current_spectral_radius > 1e-9:
                    W_res_scaled = W_res_ * (self.spectral_radius / current_spectral_radius)
                else:
                    W_res_scaled = W_res_
            except Exception:
                W_res_scaled = W_res_
        else:
            W_res_scaled = W_res_
        return W_res_scaled

    def _update_reservoir_state(self, current_input, previous_reservoir_state):
        """
        Computes one step of the reservoir state update using the correct matrix multiplication order.
        r(t) = (1-lr)*r(t-1) + lr*activation( W_in @ x(t) + W_res @ r(t-1) + bias_res )
        """
        input_contrib = torch.matmul(self.Win, current_input.T).T
        reservoir_contrib = torch.matmul(self.W_res, previous_reservoir_state.T).T
        pre_activation = input_contrib + reservoir_contrib + self.bias_res
        activated_state = self.activation_func(pre_activation)
        new_reservoir_state = (1 - self.leak_rate) * previous_reservoir_state + self.leak_rate * activated_state
        return new_reservoir_state
    
    def forward(self, input_sequence, initial_reservoir_state=None):
        batch_size, sequence_length, _ = input_sequence.shape
        device = input_sequence.device
        
        if initial_reservoir_state is None:
            current_reservoir_state = torch.zeros(batch_size, self.reservoir_dim, device=device)
        else:
            current_reservoir_state = initial_reservoir_state.to(device)

        # Since seq_length is 1 for this app, this loop runs only once.
        for t in range(sequence_length):
            current_input_t = input_sequence[:, t, :]
            current_reservoir_state = self._update_reservoir_state(current_input_t, current_reservoir_state)
            
        readout_input = current_reservoir_state
        if self.readout_uses_input:
            readout_input = torch.cat((current_reservoir_state, current_input_t), dim=1)
        
        final_output = self.W_out_layer(readout_input)
        
        return final_output, current_reservoir_state

    def reset_state(self):
        self.current_reservoir_state = None
        self.is_initial_state = True
        
    def load_weights_from_data(self, Win_data, W_res_data, bias_res_data, W_out_data, bias_out_data):
        if Win_data is not None:
            self.Win = nn.Parameter(torch.tensor(Win_data, dtype=torch.float32), requires_grad=False)
        else:
            Win_ = torch.rand(self.reservoir_dim, self.input_dim) * 2 - 1
            self.Win = nn.Parameter(Win_ * self.input_scaling, requires_grad=False)
            
        if bias_res_data is not None:
            self.bias_res = nn.Parameter(torch.tensor(bias_res_data.flatten(), dtype=torch.float32), requires_grad=False)
        else:
            bias_res_ = torch.rand(self.reservoir_dim) * 2 - 1
            self.bias_res = nn.Parameter(bias_res_ * self.bias_scaling, requires_grad=False)
            
        if W_res_data is not None:
            self.W_res = nn.Parameter(torch.tensor(W_res_data, dtype=torch.float32), requires_grad=False)
        else:
            self.W_res = nn.Parameter(self._initialize_W_res(), requires_grad=False)

        readout_input_dim = self.reservoir_dim + (self.input_dim if self.readout_uses_input else 0)
        self.W_out_layer = nn.Linear(readout_input_dim, self.output_dim, bias=True)

        if W_out_data is not None:
            W_out_numpy = np.asarray(W_out_data)
            expected_shape = (self.output_dim, readout_input_dim)
            if W_out_numpy.shape == expected_shape:
                W_out_to_load = W_out_numpy
            elif W_out_numpy.shape == (expected_shape[1], expected_shape[0]):
                W_out_to_load = W_out_numpy.T
            else:
                raise ValueError(f"Shape mismatch for W_out_data. Expected {expected_shape} or transpose, got {W_out_numpy.shape}")
            self.W_out_layer.weight = nn.Parameter(torch.tensor(W_out_to_load, dtype=torch.float32), requires_grad=False)
        else:
            nn.init.zeros_(self.W_out_layer.weight)

        if bias_out_data is not None:
            bias_out_numpy = np.asarray(bias_out_data).flatten()
            if bias_out_numpy.shape[0] != self.output_dim:
                raise ValueError(f"Shape mismatch for bias_out_data. Expected ({self.output_dim},), got {bias_out_numpy.shape}")
            self.W_out_layer.bias = nn.Parameter(torch.tensor(bias_out_numpy, dtype=torch.float32), requires_grad=False)
        else:
            if self.W_out_layer.bias is not None:
                nn.init.zeros_(self.W_out_layer.bias)

# %%
# Cell 3: Train ReservoirPy and Instantiate PyTorch ESN
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_classification
from reservoirpy.nodes import Reservoir, Ridge

# ========== 1. Load and Prepare Data ==========
print("--- Loading and Preparing Data ---")
X_train_orig, y_train_orig = load_classification('StarLightCurves', split='train')
X_test_orig, y_test_orig = load_classification('StarLightCurves', split='test')

# Squeeze the middle dimension (n_dims=1)
X_train = X_train_orig.squeeze()
X_test = X_test_orig.squeeze()

# Flatten each time-series sample into a single feature vector
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Encode labels and convert to one-hot format
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_orig)
y_test = label_encoder.transform(y_test_orig)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
y_test_labels = np.argmax(y_test_cat, axis=1)

print(f"Flattened training data shape: {X_train_reshaped.shape}")
print(f"One-hot training labels shape: {y_train_cat.shape}")

# ========== 2. Define and Train ReservoirPy ESN Model ==========
print("\n--- Training ReservoirPy ESN Model ---")
INPUT_DIM = X_train_reshaped.shape[1]
RESERVOIR_DIM = 200
OUTPUT_DIM = y_train_cat.shape[1]

reservoir = Reservoir(units=RESERVOIR_DIM, input_dim=INPUT_DIM, sr=0.4, input_connectivity=0.8,
                      rc_connectivity=0.1, input_scaling=0.3, lr=1, seed=43)
readout = Ridge(output_dim=OUTPUT_DIM, ridge=1e-1)
esn_model = reservoir >> readout

esn_model.fit(X_train_reshaped, y_train_cat)
print("ReservoirPy ESN training complete.")

# ========== 3. Calculate ReservoirPy Model Accuracy ==========
y_pred_rp = esn_model.run(X_test_reshaped)
y_pred_labels_rp = np.argmax(y_pred_rp, axis=1)
acc_rp = np.mean(y_pred_labels_rp == y_test_labels)
rec_rp = recall_score(y_test_labels, y_pred_labels_rp, average='macro')
f1_rp  = f1_score(y_test_labels, y_pred_labels_rp, average='macro')

print(f"ReservoirPy Original Results: rp — Acc: {acc_rp*100:.2f}%, rp — Rec: {rec_rp*100:.2f}%, rp — F1: {f1_rp*100:.2f}%")

# ========== 4. Get Effective Weights from ReservoirPy ==========
print("\n--- Extracting Effective Weights from ReservoirPy Model ---")
Win_eff = (reservoir.Win.toarray())
W_res_eff = reservoir.W.toarray()
bias_res_eff = (reservoir.bias.toarray().flatten())
W_out_eff = readout.Wout
bias_out_eff = readout.bias

# ========== 5. Instantiate the PyTorchESN Model ==========
print("\n--- Instantiating PyTorchESN with Loaded Weights ---")
pytorch_esn_opt1 = PyTorchESN(
    input_dim=INPUT_DIM,
    reservoir_dim=RESERVOIR_DIM,
    output_dim=OUTPUT_DIM,
    Win_data=Win_eff,
    W_res_data=W_res_eff,
    bias_res_data=bias_res_eff,
    W_out_data=W_out_eff,
    bias_out_data=bias_out_eff,
    leak_rate=reservoir.lr,
    spectral_radius=reservoir.sr,
    readout_uses_input=False
)
pytorch_esn_opt1.eval()
print("PyTorchESN instantiated and ready for verification.")

# %%
# Cell 4: Verify PyTorch Model
# Reshape the flattened test data to (n_samples, 1, n_features) for the PyTorch model
X_test_torch = torch.tensor(X_test_reshaped, dtype=torch.float32).unsqueeze(1)

print(f"PyTorch test input shape: {X_test_torch.shape}")

# Get PyTorch ESN Predictions
pytorch_esn_opt1.eval()
pytorch_esn_opt1.reset_state()
with torch.no_grad():
    y_pred_torch, _ = pytorch_esn_opt1(X_test_torch)

y_pred_labels_torch = np.argmax(y_pred_torch.numpy(), axis=1)
acc_torch = np.mean(y_pred_labels_torch == y_test_labels)
rec_torch = recall_score(y_test_labels, y_pred_labels_torch, average='macro')
f1_torch  = f1_score(y_test_labels, y_pred_labels_torch, average='macro')

print("\n--- Verification ---")
print(f"ReservoirPy Original Results: rp — Acc: {acc_rp*100:.2f}%, rp — Rec: {rec_rp*100:.2f}%, rp — F1: {f1_rp*100:.2f}%")
print(f"Torch Results: rp — Acc: {acc_torch*100:.2f}%, rp — Rec: {rec_torch*100:.2f}%, rp — F1: {f1_torch*100:.2f}%")


if np.isclose(acc_rp, acc_torch, atol=1e-2):
    print("\nSUCCESS: Accuracies are very close. Proceeding to ONNX export.")
else:
    print("\nWARNING: Accuracies differ significantly. Please review the implementation.")

# %%
General_Batch_Size = 8

# %%
# Cell 5: Export to ONNX
ONNX_MODEL_PATH = "ecg_esn.onnx"
DUMMY_BATCH_SIZE = General_Batch_Size
DUMMY_SEQUENCE_LENGTH = 1 # Fixed sequence length of 1

pytorch_esn_opt1.eval()
pytorch_esn_opt1.to('cpu')
pytorch_esn_opt1.reset_state()

dummy_input_sequence = torch.randn(
    DUMMY_BATCH_SIZE, DUMMY_SEQUENCE_LENGTH, INPUT_DIM, device='cpu', dtype=torch.float32
)

print(f"\n--- Exporting to ONNX: {ONNX_MODEL_PATH} ---")
torch.onnx.export(
    pytorch_esn_opt1,
    (dummy_input_sequence,),
    ONNX_MODEL_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_sequence'],
    output_names=['final_output', 'final_reservoir_state'],
    dynamic_axes={
        'input_sequence': {0: 'batch_size'},
        'final_output': {0: 'batch_size'},
        'final_reservoir_state': {0: 'batch_size'}
    }
)
print("ONNX export successful.")

# %%
# Cell 6: Calibrator and Engine Building
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

class ESNCalibrator(trt.IInt8Calibrator):
    def __init__(self, calibration_data, batch_size, input_tensor_name, cache_file):
        trt.IInt8Calibrator.__init__(self)
        self.batch_size = batch_size
        self.input_tensor_name = input_tensor_name
        self.cache_file = cache_file
        self.data = np.ascontiguousarray(calibration_data.reshape(-1, 1, calibration_data.shape[1]))
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)
        self.bindings = [int(self.device_input)]
    
    def get_batch_size(self): return self.batch_size
    def get_algorithm(self): return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]: return None
        batch = self.data[self.current_index : self.current_index + self.batch_size]
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return self.bindings
        
    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f: return f.read()
        except: return None
            
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f: f.write(cache)

# --- Prepare Calibration Data ---
CALIBRATION_SAMPLES = 256
calibration_data = X_train_reshaped[:CALIBRATION_SAMPLES].astype(np.float32)

def build_engine(onnx_path, engine_path, precision_flag=None):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors): print(parser.get_error(error))
            return None
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min=(1, 1, INPUT_DIM), opt=(General_Batch_Size, 1, INPUT_DIM), max=(2048, 1, INPUT_DIM))
    config.add_optimization_profile(profile)

    if precision_flag == 'INT8':
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = ESNCalibrator(calibration_data, batch_size=General_Batch_Size, input_tensor_name=input_name, cache_file=f"{engine_path}.cache")
        config.int8_calibrator = calibrator
    elif precision_flag == 'INT4':
        config.set_flag(trt.BuilderFlag.INT4)
        calibrator = ESNCalibrator(calibration_data, batch_size=General_Batch_Size, input_tensor_name=input_name, cache_file=f"{engine_path}.cache")
        config.int8_calibrator = calibrator
    elif precision_flag == 'TF32' and builder.platform_has_tf32:
        config.set_flag(trt.BuilderFlag.TF32)

    print(f"Building {precision_flag or 'TF32'} engine for {engine_path}...")
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_path}")

# Build all engines
build_engine(ONNX_MODEL_PATH, "ecg_esn_tf32.engine", 'TF32')
build_engine(ONNX_MODEL_PATH, "ecg_esn_int8.engine", 'INT8')
build_engine(ONNX_MODEL_PATH, "ecg_esn_int4.engine", 'INT4')

# %%
# Cell 8 (Final Version): Batched Inference with Integrated Performance and Power Reporting

import os
import time
import timeit
import math
import subprocess
import threading
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
# import matplotlib.pyplot as plt

# ADD THIS NEW CLASS IN PLACE OF THE OLD ONE

# The new, corrected JtopPowerMonitor class

import threading
import time
import numpy as np
from jtop import jtop

class JtopPowerMonitor:
    """
    A more robust thread-based power monitor for NVIDIA Jetson devices.
    It intelligently finds the GPU power rail and returns all metrics correctly.
    """
    def __init__(self, interval_sec=0.05):
        self.interval = interval_sec
        self.power_readings_watts = []
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.jetson = None
        self._gpu_power_key = None # To store the correct key for GPU power

    def _find_gpu_power_key(self):
        """Finds the correct dictionary key for GPU power."""
        if 'GPU' in self.jetson.power['rail']:
            return 'GPU'
        # Fallback for other models (like some Orin versions)
        for key in self.jetson.power['rail']:
            if 'GPU' in key:
                print(f"Info: Found GPU power rail key: '{key}'")
                return key
        return None

    def _monitor_power_loop(self):
        """The main loop for the monitoring thread."""
        print("Power monitor thread started...")
        while not self._stop_event.is_set():
            try:
                # Use the pre-determined key to get the power reading
                power = self.jetson.power['rail'][self._gpu_power_key]['power'] / 1000.0
                self.power_readings_watts.append(power)
            except Exception as e:
                # This warning will now be more specific if it still fails
                print(f"Warning: Could not read power from jtop key '{self._gpu_power_key}'. Error: {e}")
            time.sleep(self.interval)
        print("Power monitor thread stopped.")

    def start(self):
        """Starts the power monitoring background thread."""
        print("Starting jtop power monitor...")
        self.power_readings_watts = []
        self._stop_event.clear()
        
        try:
            self.jetson = jtop()
            self.jetson.start()
        except Exception as e:
            raise RuntimeError(f"Failed to start jtop. Is the service running? Error: {e}")

        # Find the correct power key before starting the thread
        self._gpu_power_key = self._find_gpu_power_key()
        if self._gpu_power_key is None:
            self.jetson.close()
            raise RuntimeError("Could not find a valid GPU power rail in jtop data.")

        self._monitor_thread = threading.Thread(target=self._monitor_power_loop)
        self._monitor_thread.start()

    def stop(self):
        """Stops the monitoring thread and returns the average power and all readings."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        
        if self.jetson:
            self.jetson.close()
        
        print("Power monitor stopped.")
        
        avg_power = np.mean(self.power_readings_watts) if self.power_readings_watts else 0.0
        
        # <<< CORRECTION 2: Return a tuple to match the calling code >>>
        return avg_power, self.power_readings_watts

# =====================================================================================
# PART 2: YOUR EXISTING, WORKING INFERENCE CODE (Unchanged)
# =====================================================================================
# Helper class remains the same
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, name):
        self.host = host_mem
        self.device = device_mem
        self.name = name

# Your run_inference function for a single batch remains the same
def run_inference(engine_file_path, input_data_batch):
    if not os.path.exists(engine_file_path):
        print(f"ERROR: Engine file not found at {engine_file_path}.")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    
    input_shape_for_this_run = input_data_batch.shape
    
    input_tensor_name = None
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_tensor_name = tensor_name
            break
    if input_tensor_name is None: raise RuntimeError("Input tensor not found.")
    
    context.set_input_shape(input_tensor_name, input_shape_for_this_run)
    
    inputs, outputs, stream = [], [], cuda.Stream()
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        size = trt.volume(shape) * np.dtype(dtype).itemsize
        
        host_mem = cuda.pagelocked_empty(int(size), dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        context.set_tensor_address(tensor_name, int(device_mem))
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem, tensor_name))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, tensor_name))

    # input_host_buffer = inputs[0].host
    # np.copyto(input_host_buffer, input_data_batch.ravel())

    # cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
    # context.execute_async_v3(stream_handle=stream.handle)
    # for out in outputs:
    #     cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # stream.synchronize()

    input_host_buffer = inputs[0].host
    input_host_buffer.fill(0)
    input_host_buffer[:input_data_batch.size] = input_data_batch.ravel()

    cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    stream.synchronize()

    results = {}
    for out in outputs:
        actual_output_shape = context.get_tensor_shape(out.name)
        num_elements = trt.volume(actual_output_shape)
        valid_output_slice = out.host[:num_elements]
        results[out.name] = valid_output_slice.reshape(actual_output_shape)
        
    return results

# In your final cell (e.g., Cell 8), ADD THIS NEW FUNCTION after your `run_inference` function.

# In your final cell, ADD THIS NEW FUNCTION

def benchmark_single_sample_latency(engine_file_path, single_input_sample):
    """
    Performs a dedicated benchmark to find the latency of a single sample (batch_size=1).
    This uses CUDA events for precise GPU execution time measurement.

    Args:
        engine_file_path (str): Path to the serialized .engine file.
        single_input_sample (np.ndarray): A single input sample, shaped (1, 1, INPUT_DIM).

    Returns:
        float: The average single-sample latency in milliseconds.
    """
    print(f"--- Benchmarking single-sample latency for: {os.path.basename(engine_file_path)} ---")
    
    # Standard TensorRT setup
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        if not engine:
            print(f"ERROR: Failed to deserialize engine {engine_file_path}.")
            return 0.0

    context = engine.create_execution_context()
    
    # Set the context shape for a single sample
    input_shape = single_input_sample.shape
    input_tensor_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
    context.set_input_shape(input_tensor_name, input_shape)

    # Allocate buffers
    inputs, outputs, stream = [], [], cuda.Stream()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(trt.volume(shape))
        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        context.set_tensor_address(name, int(device_mem))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem, name))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, name))

    # Copy the single sample data to the host buffer
    np.copyto(inputs[0].host, single_input_sample.ravel())

    # Warm-up runs
    for _ in range(50):
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    # Timed runs for latency measurement
    gpu_times = []
    start_event = cuda.Event()
    stop_event = cuda.Event()

    for _ in range(100):
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
        start_event.record(stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stop_event.record(stream)
        stream.synchronize()
        gpu_times.append(stop_event.time_since(start_event))

    print("Single-sample latency benchmark complete.")
    return np.mean(gpu_times)
# =====================================================================================
# PART 3: MAIN EXECUTION LOGIC (Your code, now wrapped with monitoring)
# =====================================================================================

# --- Define the engine files you built ---
engines_to_run = {
    # Using your original names
    "TF32 Engine": "ecg_esn_tf32.engine",
    "INT8 Engine": "ecg_esn_int8.engine",
    "INT4 Engine": "ecg_esn_int4.engine"
}

# --- Prepare your FULL test input data ---
full_test_input_np = X_test_reshaped.reshape(-1, 1, INPUT_DIM).astype(np.float32)
num_test_samples = full_test_input_np.shape[0]

# --- Define a manageable batch size for inference ---
INFERENCE_BATCH_SIZE = General_Batch_Size 

# --- Loop through each engine, run batched inference, and get its output ---
# This dictionary will store the final stitched output arrays
all_engine_outputs = {}
# This list will store the final report data for the table
report_data = []

# Initialize the power monitor once
power_monitor = JtopPowerMonitor()

for engine_name, engine_path in engines_to_run.items():
    
    all_batch_outputs = []
    num_batches = math.ceil(num_test_samples / INFERENCE_BATCH_SIZE)
    
    print(f"\n--- Running Batched Inference for: {engine_name} ---")
    
    # *** START of Monitoring Block ***
    power_monitor.start()
    time.sleep(0.5) # Let monitor stabilize
    total_start_time = timeit.default_timer()

    for i in range(num_batches):
        start_idx = i * INFERENCE_BATCH_SIZE
        end_idx = min((i + 1) * INFERENCE_BATCH_SIZE, num_test_samples)
        batch_input_data = full_test_input_np[start_idx:end_idx]
        batch_results = run_inference(engine_path, batch_input_data)
        
        if batch_results:
            all_batch_outputs.append(batch_results['final_output'])
        else:
            print(f"Inference failed for a batch in {engine_name}. Aborting.")
            all_batch_outputs = []
            break
            
    # *** END of Monitoring Block ***
    total_end_time = timeit.default_timer()
    avg_power, _ = power_monitor.stop()

    if all_batch_outputs:
        single_sample_input = full_test_input_np[0:1] # Prepare one sample
        single_latency = benchmark_single_sample_latency(engine_path, single_sample_input)
        # --- Calculate Metrics for this Engine ---
        total_latency_s = total_end_time - total_start_time
        throughput = num_test_samples / total_latency_s
        energy = avg_power * total_latency_s
        
        # Stitch the results from all batches back together
        full_output = np.vstack(all_batch_outputs)
        all_engine_outputs[engine_name] = full_output
        print(f"Successfully aggregated results. Final output shape: {full_output.shape}")

        # Calculate accuracy for this engine's full output
        y_pred = np.argmax(full_output, axis=1)
        acc = np.mean(y_pred == y_test_labels)
        rec = recall_score(y_test_labels, y_pred, average='macro')
        f1  = f1_score(y_test_labels, y_pred, average='macro')

        # Store all metrics for the final report
        report_data.append({
            "Engine": engine_name,
            "Accuracy": acc * 100,
            "Recall": rec * 100,
            "F1_Score": f1 * 100,
            "Latency": total_latency_s * 1000,
            "Single_Sample_Latency_ms": single_latency,
            "Throughput": throughput,
            "Avg Power": avg_power,
            "Energy": energy
        })

# =====================================================================================
# FINAL REPORT TABLE (Updated with the new column)
# =====================================================================================
print("\n" + "="*125)
print("--- FINAL PERFORMANCE AND POWER REPORT (JETSON) ---")
print(f"--- Full Test Set: {num_test_samples} samples, Batched Throughput Test Batch Size: {INFERENCE_BATCH_SIZE} ---")
print("="*125)

header = [
    "Engine", "Accuracy (%)", "Recall (%)", "F1_Score (%)", "Per-Sample Latency (ms)", "Batched Latency (ms)",
    "Throughput (samples/s)", "Avg Power (W)", "Energy (J)"
]
print(f"{header[0]:<16} | {header[1]:<15} | {header[2]:<15} | {header[3]:<15} | {header[4]:<25} | {header[5]:<20} | {header[6]:<25} | {header[7]:<15} | {header[8]:<12}")
print("-" * 125)

print(f"{'ReservoirPy':<16} | {acc_rp*100:<15.2f} |{rec_rp*100:<15.2f} |{f1_rp*100:<15.2f} | {'N/A':<25} | {'N/A':<20} | {'N/A':<25} | {'N/A':<15} | {'N/A':<12}")
print(f"{'PyTorch':<16} | {acc_torch*100:<15.2f} | {rec_torch*100:<15.2f} | {f1_torch*100:<15.2f} | {'N/A':<25} | {'N/A':<20} | {'N/A':<25} | {'N/A':<15} | {'N/A':<12}")

for data in report_data:
    print(
        f"{data['Engine']:<16} | "
        f"{data['Accuracy']:<15.2f} | "
        f"{data['Recall']:<15.2f} | "
        f"{data['F1_Score']:<15.2f} | "
        f"{data['Single_Sample_Latency_ms']:<25.4f} | "  # <-- NEW COLUMN IN REPORT
        f"{data['Latency']:<20.4f} | " # This is the time for the whole batch job
        f"{data['Throughput']:<25,.2f} | "
        f"{data['Avg Power']:<15.4f} | "
        f"{data['Energy']:.4f}"
    )
print("-" * 125)
print("\nExcel/TSV Table (copy and paste into Excel):")
header = ["Engine", "Accuracy (%)", "Recall (%)", "F1_Score (%)", "Total Latency (ms)", "Single_Sample_Latency_ms", "Throughput (samples/s)", "Avg Power (W)", "Energy (J)"]
print('\t'.join(header))
for data in report_data:
    row = [
        data['Engine'],
        f"{data['Accuracy']:.2f}",
        f"{data['Recall']:.2f}",
        f"{data['F1_Score']:.2f}",
        f"{data['Latency']:.4f}",
        f"{data['Single_Sample_Latency_ms']:.4f}",
        f"{data['Throughput']:.2f}",
        f"{data['Avg Power']:.4f}",
        f"{data['Energy']:.4f}"
    ]
    print('\t'.join(row))

