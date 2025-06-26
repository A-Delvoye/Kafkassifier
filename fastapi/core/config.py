import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "t0nVra1SecretClé-!à-chang3r")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 90