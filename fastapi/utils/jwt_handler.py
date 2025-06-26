from datetime import datetime, timedelta, timezone
import jwt
from core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from models.user import User
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select
from db.session import get_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_user(
    token: str = Security(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
        )
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Utilisateur non identifié")

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    return user

def create_access_token(data: dict) -> str:
    # access_token = create_access_token({"sub": str(user.id)})
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if "sub" not in payload:
            return None
        return payload
    except jwt.PyJWTError:
        return None