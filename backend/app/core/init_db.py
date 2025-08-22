import logging
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine
from app.models.database import Base, User
from app.core.auth import AuthService
from app.core.config import settings

logger = logging.getLogger(__name__)

def init_db() -> None:
    """
    初始化数据库
    创建表和初始数据
    """
    try:
        # 创建所有表
        logger.info("创建数据库表...")
        Base.metadata.create_all(bind=engine)
        
        # 创建初始用户
        db = SessionLocal()
        try:
            create_initial_data(db)
        finally:
            db.close()
            
        logger.info("数据库初始化完成")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise

def create_initial_data(db: Session) -> None:
    """
    创建初始数据
    """
    # 检查是否已存在管理员用户
    admin_user = db.query(User).filter(User.email == settings.first_superuser_email).first()
    
    if not admin_user:
        logger.info("创建初始管理员用户...")
        
        # 创建管理员用户
        admin_user = User(
            email=settings.first_superuser_email,
            username="admin",
            full_name="系统管理员",
            password_hash=AuthService.get_password_hash(settings.first_superuser_password),
            role="admin",
            is_active=True,
            is_superuser=True
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        logger.info(f"管理员用户创建成功: {admin_user.email}")
    else:
        logger.info("管理员用户已存在")
    
    # 可以在这里添加其他初始数据
    # 例如：默认文档类别、系统配置等
    
def create_sample_data(db: Session) -> None:
    """
    创建示例数据（可选）
    """
    try:
        # 创建示例用户
        sample_users = [
            {
                "email": "user1@example.com",
                "username": "user1",
                "full_name": "普通用户1",
                "password": "password123",
                "role": "user"
            },
            {
                "email": "manager1@example.com",
                "username": "manager1",
                "full_name": "管理员1",
                "password": "password123",
                "role": "manager"
            }
        ]
        
        for user_data in sample_users:
            existing_user = db.query(User).filter(User.email == user_data["email"]).first()
            if not existing_user:
                user = User(
                    email=user_data["email"],
                    username=user_data["username"],
                    full_name=user_data["full_name"],
                    hashed_password=get_password_hash(user_data["password"]),
                    role=user_data["role"],
                    is_active=True,
                    is_verified=True
                )
                db.add(user)
        
        db.commit()
        logger.info("示例数据创建完成")
        
    except Exception as e:
        logger.error(f"创建示例数据失败: {e}")
        db.rollback()
        raise

if __name__ == "__main__":
    # 直接运行此脚本来初始化数据库
    logging.basicConfig(level=logging.INFO)
    init_db()