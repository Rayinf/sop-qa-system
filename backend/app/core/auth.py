from datetime import datetime, timedelta
from typing import Optional, Union, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy.orm import Session
import secrets
import logging
from app.core.config import settings
from app.core.database import get_db
from app.models.database import User
from app.models.schemas import TokenData

logger = logging.getLogger(__name__)

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    scheme_name="JWT"
)

# 可选的OAuth2 scheme
optional_oauth2_scheme = HTTPBearer(auto_error=False)

class AuthService:
    """认证服务类"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """生成密码哈希"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(
        subject: Union[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.access_token_expire_minutes
            )
        
        to_encode = {
            "exp": expire,
            "sub": str(subject),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(
        subject: Union[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建刷新令牌"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=settings.refresh_token_expire_days
            )
        
        to_encode = {
            "exp": expire,
            "sub": str(subject),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
        """验证令牌"""
        try:
            payload = jwt.decode(
                token, 
                settings.secret_key, 
                algorithms=[settings.algorithm]
            )
            
            # 检查令牌类型
            if payload.get("type") != token_type:
                return None
            
            username: str = payload.get("sub")
            if username is None:
                return None
            
            token_data = TokenData(username=username)
            return token_data
            
        except JWTError as e:
            logger.error(f"JWT验证失败: {e}")
            return None
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """用户认证"""
        try:
            # 查找用户
            user = db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                return None
            
            # 验证密码
            if not AuthService.verify_password(password, user.password_hash):
                return None
            
            # 检查用户状态
            if not user.is_active:
                return None
            
            # 更新最后登录时间
            user.last_login = datetime.utcnow()
            db.commit()
            
            return user
            
        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        try:
            return db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    @staticmethod
    def create_user(db: Session, username: str, email: str, password: str, 
                   full_name: Optional[str] = None, role: str = "operator") -> Optional[User]:
        """创建用户"""
        try:
            # 检查用户名和邮箱是否已存在
            existing_user = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                return None
            
            # 创建新用户
            hashed_password = AuthService.get_password_hash(password)
            user = User(
                username=username,
                email=email,
                password_hash=hashed_password,
                full_name=full_name,
                role=role,
                is_active=True
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"用户创建成功: {username}")
            return user
            
        except Exception as e:
            logger.error(f"用户创建失败: {e}")
            db.rollback()
            return None
    
    @staticmethod
    def generate_api_key() -> str:
        """生成API密钥"""
        return secrets.token_urlsafe(32)

# 依赖注入函数
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 验证令牌
        token_data = AuthService.verify_token(token, "access")
        if token_data is None:
            raise credentials_exception
        
        # 获取用户
        user = AuthService.get_user_by_username(db, token_data.username)
        if user is None:
            raise credentials_exception
        
        # 检查用户状态
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="用户账户已被禁用"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取当前用户失败: {e}")
        raise credentials_exception

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户账户已被禁用"
        )
    return current_user

async def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前超级用户
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要超级用户权限"
        )
    return current_user

async def get_current_user_optional(
    credentials = Depends(optional_oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    获取当前用户（可选）
    如果token无效或不存在，返回None而不是抛出异常
    """
    if not credentials:
        return None
    
    try:
        token_data = AuthService.verify_token(credentials.credentials)
        if not token_data:
            return None
        
        user = AuthService.get_user_by_username(db, username=token_data.username)
        if not user:
            return None
        
        return user
    except Exception as e:
        logger.debug(f"可选认证失败: {e}")
        return None

# 权限检查函数
def create_role_checker(required_roles):
    """创建角色权限检查函数"""
    # 如果传入的是字符串，转换为列表
    if isinstance(required_roles, str):
        required_roles = [required_roles]
    
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in required_roles and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要以下角色权限之一: {', '.join(required_roles)}"
            )
        return current_user
    return role_checker

# 便捷的权限检查函数
def require_role(required_roles):
    """角色权限检查依赖项"""
    return Depends(create_role_checker(required_roles))

def require_roles(required_roles: list):
    """多角色权限检查装饰器"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in required_roles and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要以下角色之一: {', '.join(required_roles)}"
            )
        return current_user
    return role_checker

# 权限常量
class Permissions:
    """权限常量类"""
    UPLOAD_DOCUMENT = "upload_document"
    DELETE_DOCUMENT = "delete_document"
    VIEW_DOCUMENT = "view_document"
    ASK_QUESTION = "ask_question"
    VIEW_LOGS = "view_logs"
    MANAGE_USERS = "manage_users"
    ADMIN_ACCESS = "admin_access"

# 角色权限映射
ROLE_PERMISSIONS = {
    "admin": [
        Permissions.UPLOAD_DOCUMENT,
        Permissions.DELETE_DOCUMENT,
        Permissions.VIEW_DOCUMENT,
        Permissions.ASK_QUESTION,
        Permissions.VIEW_LOGS,
        Permissions.MANAGE_USERS,
        Permissions.ADMIN_ACCESS
    ],
    "engineer": [
        Permissions.UPLOAD_DOCUMENT,
        Permissions.VIEW_DOCUMENT,
        Permissions.ASK_QUESTION,
        Permissions.VIEW_LOGS
    ],
    "operator": [
        Permissions.VIEW_DOCUMENT,
        Permissions.ASK_QUESTION
    ],
    "viewer": [
        Permissions.VIEW_DOCUMENT
    ]
}

def check_permission(user: User, permission: str) -> bool:
    """检查用户权限"""
    if user.is_superuser:
        return True
    
    user_permissions = ROLE_PERMISSIONS.get(user.role, [])
    return permission in user_permissions

def require_permission(permission: str):
    """权限检查装饰器"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少权限: {permission}"
            )
        return current_user
    return permission_checker