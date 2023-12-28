from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import HongosModel
import uvicorn



#Se declara la API y se importa el modelo del archivo model.py
app = FastAPI()
modelo_hongos = HongosModel()


#Se especifica y valida el tipo de dato tanto del input como del output del usuario
class InputData(BaseModel):
    cap_shape: str
    cap_surface: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    habitat: str

class OutputData(BaseModel):
    prediction: int
    probability: float


#Se declara la ruta de ejecución de la API
@app.post("/predict", response_model=OutputData)
async def predict(data: InputData):

    #Se especifica la ejecución de la API al ejecutar model_prediction
    try:
        prediction, probability = modelo_hongos.model_prediction(**data.dict())
        return {"prediction": prediction, "probability": probability}

    #Alerta de mensaje de error por si algo sale mal con la función model_prediction de la clase del modelo
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


#Ejemplos de input de usuario extraídos del dataset importado para la API para poder probar la eficacia de la API de ML
'''
VENENOSO=1
{
  "cap_shape": "x",
  "cap_surface": "s",
  "bruises": "t",
  "odor": "p",
  "gill_attachment": "f",
  "gill_spacing": "c",
  "veil_type": "p",
  "veil_color": "w",
  "ring_number": "o",
  "ring_type": "p",
  "spore_print_color": "k",
  "habitat": "u"
}

{
  "cap_shape": "f",
  "cap_surface": "s",
  "bruises": "t",
  "odor": "p",
  "gill_attachment": "f",
  "gill_spacing": "c",
  "veil_type": "p",
  "veil_color": "w",
  "ring_number": "o",
  "ring_type": "p",
  "spore_print_color": "n",
  "habitat": "g"
}



#==========================================



NO VENENOSO=0
{
  "cap_shape": "x",
  "cap_surface": "s",
  "bruises": "t",
  "odor": "a",
  "gill_attachment": "f",
  "gill_spacing": "c",
  "veil_type": "p",
  "veil_color": "w",
  "ring_number": "o",
  "ring_type": "p",
  "spore_print_color": "n",
  "habitat": "g"
}

{
  "cap_shape": "b",
  "cap_surface": "y",
  "bruises": "t",
  "odor": "l",
  "gill_attachment": "f",
  "gill_spacing": "c",
  "veil_type": "p",
  "veil_color": "w",
  "ring_number": "o",
  "ring_type": "p",
  "spore_print_color": "k",
  "habitat": "m"
}
'''
