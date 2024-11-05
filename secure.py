from fastapi import APIRouter, Depends
from . import auth

router = APIRouter()

@router.get("/")
async def get_testroute(user: dict = Depends(auth.get_user)):
    return user
