import joblib

model = joblib.load(
    "models/hospital_readmission_model.pkl"
)

print("\n====================================")
print("MODEL TYPE")
print("====================================")

print(type(model))


print("\n====================================")
print("PIPELINE STEPS")
print("====================================")

if hasattr(model, "steps"):
    for name, step in model.steps:
        print(name, "->", type(step))


print("\n====================================")
print("MODEL PARAMETERS")
print("====================================")

print(model)


print("\n====================================")
print("FEATURE NAMES")
print("====================================")

if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)

else:
    print("feature_names_in_ not available")


print("\n====================================")
print("NUMBER OF FEATURES")
print("====================================")

if hasattr(model, "n_features_in_"):
    print(model.n_features_in_)

else:
    print("n_features_in_ not available")