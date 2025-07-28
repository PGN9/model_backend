import onnxruntime as ort
print(hasattr(ort, 'InferenceSession'))
print(ort.__file__)