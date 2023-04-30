from fastapi import APIRouter

router = APIRouter()

@router.get(
        "/", 
        tags=['Index'],
        description='Returns a welcome message'
    )
async def root()->dict:
    """Returns a welcome message

    Returns:
        dict: returns a welcome message
    """
    return {"message": "Welcome to Stock/EFT Prediction"}
