# import jwt
# from core.config import SECRET_KEY, ALGORITHM
# from fastapi.security import OAuth2PasswordBearer
# from fastapi import Security


# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# def get_user(token: str = Security(oauth2_scheme)):

#     payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#     return (payload["sub"])