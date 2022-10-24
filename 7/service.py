import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    country: str
    rating: float

# coolmodel
# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
# coolmodel2
model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
# dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()
svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(UserProfile):
    # vector = dv.transform(application_data)

    prediction = await model_runner.predict.async_run(UserProfile)
    result = prediction[0]

    return {"result": result}
