from fastapi import FastAPI, Depends
from . import secure, prediction, public, auth

app = FastAPI()

app.include_router(
    public.router,
    prefix="/api/v1/public"
)
app.include_router(
    secure.router,
    prefix="/api/v1/secure",
    dependencies=[Depends(auth.get_user)]
)

app.include_router(
	prediction.router,
	prefix = "/api/v1/inference",
	dependencies = [Depends(auth.get_user)]
)