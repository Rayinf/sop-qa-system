import { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  Button,
  Avatar,
  Upload,
  Typography,
  Space,
  Divider,
  Row,
  Col,
  Statistic,
  List,
  Tag,
  message,
  Modal,
  Switch
} from 'antd'
import {
  UserOutlined,
  EditOutlined,
  CameraOutlined,
  LockOutlined,
  SettingOutlined,
  QuestionCircleOutlined,
  FileTextOutlined,
  TrophyOutlined
} from '@ant-design/icons'
import type { UploadProps } from 'antd'
import { User } from '../types/index'
import { StatisticsApi } from '../services/statisticsApi'
import { authApi } from '../services/authApi'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input

interface ExtendedUser extends User {
  department?: string;
  position?: string;
  phone?: string;
  bio?: string;
  avatar_url?: string;
}

interface ProfileProps {
  user?: ExtendedUser
}

const Profile = ({ user }: ProfileProps) => {
  const [loading, setLoading] = useState(false)
  const [editMode, setEditMode] = useState(false)
  const [passwordVisible, setPasswordVisible] = useState(false)
  const [userStats, setUserStats] = useState<{
    total_questions: number;
    total_documents_uploaded: number;
    avg_feedback_score: number;
    total_login_days: number;
    last_login: string;
  } | null>(null)
  const [recentActivity, setRecentActivity] = useState<{
    id: number;
    type: string;
    content: string;
    time: string;
  }[]>([])
  const [form] = Form.useForm()
  const [passwordForm] = Form.useForm()

  useEffect(() => {
    // 加载用户统计数据
    const loadUserStats = async () => {
      try {
        const personalStats = await StatisticsApi.getPersonalQAStatistics()
        
        // 转换API数据格式以匹配UI需求
         const stats = {
           total_questions: personalStats.total_questions,
           total_documents_uploaded: 0, // 暂时设为0，后续可以添加文档上传统计API
           avg_feedback_score: personalStats.feedback_distribution ? 
             Object.entries(personalStats.feedback_distribution).reduce((acc, [score, count]) => 
               acc + (parseInt(score) * count), 0) / 
             Object.values(personalStats.feedback_distribution).reduce((acc, count) => acc + count, 0) || 0 : 0,
           total_login_days: 0, // 暂时设为0，后续可以添加登录统计API
           last_login: user?.last_login || ''
         }
        
        // 从最近的QA记录生成活动数据
        const recentQA = personalStats.recent_questions || []
        const activity = recentQA.map((qa, index) => ({
          id: index + 1,
          type: 'question',
          content: `询问了：${qa.question.substring(0, 30)}${qa.question.length > 30 ? '...' : ''}`,
          time: qa.created_at
        }))
        
        // 添加一些默认活动项（如果API数据不足）
        if (activity.length < 4) {
          activity.push(
            {
              id: activity.length + 1,
              type: 'feedback',
              content: '对问答结果给出了评价',
              time: new Date().toISOString()
            }
          )
        }
        
        setUserStats(stats)
        setRecentActivity(activity)
      } catch (error) {
        console.error('加载用户数据失败:', error)
        // 使用默认数据作为后备
        setUserStats({
          total_questions: 0,
          total_documents_uploaded: 0,
          avg_feedback_score: 0,
          total_login_days: 0,
          last_login: user?.last_login || ''
        })
        setRecentActivity([])
      }
    }
    
    loadUserStats()
    
    // 初始化表单数据
    if (user) {
      form.setFieldsValue({
        full_name: user.full_name,
        email: user.email,
        department: user.department,
        position: user.position,
        phone: user.phone,
        bio: user.bio
      })
    }
  }, [user, form])

  // 头像上传配置
  const avatarUploadProps: UploadProps = {
    name: 'avatar',
    showUploadList: false,
    beforeUpload: (file: File) => {
      const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png'
      if (!isJpgOrPng) {
        message.error('只能上传 JPG/PNG 格式的图片')
        return false
      }
      const isLt2M = file.size / 1024 / 1024 < 2
      if (!isLt2M) {
        message.error('图片大小不能超过 2MB')
        return false
      }
      return true
    },
    onChange: (info: any) => {
      if (info.file.status === 'done') {
        message.success('头像上传成功')
      } else if (info.file.status === 'error') {
        message.error('头像上传失败')
      }
    }
  }

  // 保存个人信息
  const handleSaveProfile = async () => {
    try {
      setLoading(true)
      const values = await form.validateFields()
      
      // 调用真实API更新用户信息
      await authApi.updateCurrentUser({
        full_name: values.full_name,
        email: values.email,
        // 注意：department, position, phone, bio 等字段可能需要后端支持
      })
      
      message.success('个人信息更新成功')
      setEditMode(false)
    } catch (error) {
      console.error('更新个人信息失败:', error)
      message.error('更新失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  // 修改密码
  const handleChangePassword = async () => {
    try {
      setLoading(true)
      const values = await passwordForm.validateFields()
      
      // 调用真实的密码修改API
      await authApi.changePassword(values.current_password, values.new_password)
      
      message.success('密码修改成功')
      setPasswordVisible(false)
      passwordForm.resetFields()
    } catch (error: any) {
      console.error('密码修改失败:', error)
      message.error(error.message || '密码修改失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  // 格式化时间
  const formatTime = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const hours = Math.floor(diff / (1000 * 60 * 60))
    
    if (hours < 1) {
      const minutes = Math.floor(diff / (1000 * 60))
      return `${minutes}分钟前`
    } else if (hours < 24) {
      return `${hours}小时前`
    } else {
      const days = Math.floor(hours / 24)
      return `${days}天前`
    }
  }

  // 获取活动图标
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'question':
        return <QuestionCircleOutlined style={{ color: '#1890ff' }} />
      case 'upload':
        return <FileTextOutlined style={{ color: '#52c41a' }} />
      case 'feedback':
        return <TrophyOutlined style={{ color: '#faad14' }} />
      default:
        return <UserOutlined style={{ color: '#666' }} />
    }
  }

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>个人资料</Title>
        <Text type="secondary">
          管理您的个人信息和账户设置。
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 左侧：个人信息 */}
        <Col xs={24} lg={16}>
          <Card
            title="基本信息"
            extra={
              <Button
                type={editMode ? 'default' : 'primary'}
                icon={<EditOutlined />}
                onClick={() => setEditMode(!editMode)}
              >
                {editMode ? '取消编辑' : '编辑资料'}
              </Button>
            }
          >
            <Row gutter={[24, 0]}>
              {/* 头像区域 */}
              <Col xs={24} sm={8} style={{ textAlign: 'center' }}>
                <div style={{ position: 'relative', display: 'inline-block' }}>
                  <Avatar
                    size={120}
                    src={user?.avatar_url}
                    icon={<UserOutlined />}
                    style={{ marginBottom: 16 }}
                  />
                  {editMode && (
                    <Upload {...avatarUploadProps}>
                      <Button
                        type="primary"
                        shape="circle"
                        icon={<CameraOutlined />}
                        size="small"
                        style={{
                          position: 'absolute',
                          bottom: 16,
                          right: 0,
                          border: '2px solid white'
                        }}
                      />
                    </Upload>
                  )}
                </div>
                <div>
                  <Title level={4} style={{ margin: 0 }}>
                    {user?.full_name || '未设置姓名'}
                  </Title>
                  <Text type="secondary">{user?.position || '未设置职位'}</Text>
                </div>
              </Col>

              {/* 表单区域 */}
              <Col xs={24} sm={16}>
                <Form
                  form={form}
                  layout="vertical"
                  disabled={!editMode}
                >
                  <Row gutter={16}>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="full_name"
                        label="姓名"
                        rules={[{ required: true, message: '请输入姓名' }]}
                      >
                        <Input placeholder="请输入姓名" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="email"
                        label="邮箱"
                        rules={[
                          { required: true, message: '请输入邮箱' },
                          { type: 'email', message: '请输入有效的邮箱地址' }
                        ]}
                      >
                        <Input placeholder="请输入邮箱" disabled />
                      </Form.Item>
                    </Col>
                  </Row>
                  
                  <Row gutter={16}>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="department"
                        label="部门"
                      >
                        <Input placeholder="请输入部门" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="position"
                        label="职位"
                      >
                        <Input placeholder="请输入职位" />
                      </Form.Item>
                    </Col>
                  </Row>
                  
                  <Row gutter={16}>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="phone"
                        label="电话"
                      >
                        <Input placeholder="请输入电话号码" />
                      </Form.Item>
                    </Col>
                  </Row>
                  
                  <Form.Item
                    name="bio"
                    label="个人简介"
                  >
                    <TextArea
                      rows={3}
                      placeholder="请输入个人简介"
                      maxLength={200}
                      showCount
                    />
                  </Form.Item>
                </Form>

                {editMode && (
                  <div style={{ textAlign: 'right', marginTop: 16 }}>
                    <Space>
                      <Button onClick={() => setEditMode(false)}>
                        取消
                      </Button>
                      <Button
                        type="primary"
                        loading={loading}
                        onClick={handleSaveProfile}
                      >
                        保存
                      </Button>
                    </Space>
                  </div>
                )}
              </Col>
            </Row>
          </Card>

          {/* 账户设置 */}
          <Card title="账户设置" style={{ marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <Text strong>修改密码</Text>
                  <br />
                  <Text type="secondary">定期更新密码以保护账户安全</Text>
                </div>
                <Button
                  icon={<LockOutlined />}
                  onClick={() => setPasswordVisible(true)}
                >
                  修改密码
                </Button>
              </div>
              
              <Divider />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <Text strong>邮件通知</Text>
                  <br />
                  <Text type="secondary">接收系统通知和更新提醒</Text>
                </div>
                <Switch defaultChecked />
              </div>
              
              <Divider />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <Text strong>隐私设置</Text>
                  <br />
                  <Text type="secondary">控制个人信息的可见性</Text>
                </div>
                <Button icon={<SettingOutlined />}>
                  设置
                </Button>
              </div>
            </Space>
          </Card>
        </Col>

        {/* 右侧：统计和活动 */}
        <Col xs={24} lg={8}>
          {/* 使用统计 */}
          <Card title="使用统计" style={{ marginBottom: 24 }}>
            {userStats ? (
              <Row gutter={[0, 16]}>
                <Col span={24}>
                  <Statistic
                    title="提问总数"
                    value={userStats.total_questions}
                    prefix={<QuestionCircleOutlined />}
                  />
                </Col>
                <Col span={24}>
                  <Statistic
                    title="上传文档"
                    value={userStats.total_documents_uploaded}
                    prefix={<FileTextOutlined />}
                  />
                </Col>
                <Col span={24}>
                  <Statistic
                    title="平均评分"
                    value={userStats.avg_feedback_score}
                    precision={1}
                    suffix="/ 5.0"
                    prefix={<TrophyOutlined />}
                  />
                </Col>
                <Col span={24}>
                  <Statistic
                    title="活跃天数"
                    value={userStats.total_login_days}
                    suffix="天"
                    prefix={<UserOutlined />}
                  />
                </Col>
              </Row>
            ) : (
              <Text type="secondary">加载中...</Text>
            )}
          </Card>

          {/* 最近活动 */}
          <Card title="最近活动">
            <List
              dataSource={recentActivity}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={getActivityIcon(item.type)}
                    title={
                      <Text style={{ fontSize: '14px' }}>
                        {item.content}
                      </Text>
                    }
                    description={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {formatTime(item.time)}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* 修改密码模态框 */}
      <Modal
        title="修改密码"
        open={passwordVisible}
        onOk={handleChangePassword}
        onCancel={() => {
          setPasswordVisible(false)
          passwordForm.resetFields()
        }}
        confirmLoading={loading}
        okText="确认修改"
        cancelText="取消"
      >
        <Form
          form={passwordForm}
          layout="vertical"
        >
          <Form.Item
            name="current_password"
            label="当前密码"
            rules={[{ required: true, message: '请输入当前密码' }]}
          >
            <Input.Password placeholder="请输入当前密码" />
          </Form.Item>
          
          <Form.Item
            name="new_password"
            label="新密码"
            rules={[
              { required: true, message: '请输入新密码' },
              { min: 6, message: '密码长度至少6位' }
            ]}
          >
            <Input.Password placeholder="请输入新密码" />
          </Form.Item>
          
          <Form.Item
            name="confirm_password"
            label="确认新密码"
            dependencies={['new_password']}
            rules={[
              { required: true, message: '请确认新密码' },
              ({ getFieldValue }) => ({
                validator(_: any, value: string) {
                  if (!value || getFieldValue('new_password') === value) {
                    return Promise.resolve()
                  }
                  return Promise.reject(new Error('两次输入的密码不一致'))
                }
              })
            ]}
          >
            <Input.Password placeholder="请再次输入新密码" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default Profile