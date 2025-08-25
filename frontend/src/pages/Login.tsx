import { useState } from 'react'
import {
  Form,
  Input,
  Button,
  Card,
  Typography,
  Space,
  Alert,
  Checkbox
} from 'antd'
import {
  UserOutlined,
  LockOutlined,
  EyeInvisibleOutlined,
  EyeTwoTone
} from '@ant-design/icons'
import { LoginRequest } from '../types/index'

const { Title, Text } = Typography

interface LoginProps {
  onLogin: (credentials: LoginRequest) => Promise<void>
  loading?: boolean
  error?: string
}

const Login = ({ onLogin, loading = false, error }: LoginProps) => {
  const [form] = Form.useForm()
  const [rememberMe, setRememberMe] = useState(false)

  const handleSubmit = async (values: LoginRequest) => {
    try {
      await onLogin(values)
    } catch (err) {
      // 错误处理由父组件处理
      console.error('登录失败:', err)
    }
  }

  const fillDemoAccount = () => {
    form.setFieldsValue({
      email: 'admin@example.com',
      password: 'admin123456'
    })
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    }}>
      <Card
        style={{
          width: 400,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          borderRadius: 12
        }}
        styles={{ body: { padding: '32px' } }}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              width: 64,
              height: 64,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
              color: 'white',
              fontSize: '24px',
              fontWeight: 'bold'
            }}>
              S
            </div>
            <Title level={2} style={{ margin: 0, color: '#1f2937' }}>
              langchain知识库问答系统
            </Title>
            <Text type="secondary">
              请登录您的账户
            </Text>
          </div>

          {error && (
            <Alert
              message={error}
              type="error"
              showIcon
              closable
            />
          )}

          <Form
            form={form}
            name="login"
            onFinish={handleSubmit}
            autoComplete="off"
            size="large"
          >
            <Form.Item
              name="email"
              rules={[
                {
                  required: true,
                  message: '请输入邮箱地址！',
                },
                {
                  type: 'email',
                  message: '请输入有效的邮箱地址！',
                },
              ]}
            >
              <Input
                prefix={<UserOutlined />}
                placeholder="邮箱地址"
                autoComplete="email"
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[
                {
                  required: true,
                  message: '请输入密码！',
                },
                {
                  min: 6,
                  message: '密码至少6位字符！',
                },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="密码"
                autoComplete="current-password"
                iconRender={(visible) => (visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />)}
              />
            </Form.Item>

            <Form.Item>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Checkbox
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                >
                  记住我
                </Checkbox>
                <Button type="link" style={{ padding: 0 }}>
                  忘记密码？
                </Button>
              </div>
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                style={{
                  width: '100%',
                  height: 44,
                  borderRadius: 8,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  border: 'none',
                  fontSize: '16px',
                  fontWeight: 500
                }}
              >
                {loading ? '登录中...' : '登录'}
              </Button>
            </Form.Item>
          </Form>

          <div style={{ textAlign: 'center' }}>
            <Text type="secondary">
              还没有账户？
              <Button type="link" style={{ padding: '0 4px' }}>
                立即注册
              </Button>
            </Text>
          </div>

          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <Button 
              type="link" 
              size="small"
              onClick={fillDemoAccount}
              style={{ 
                fontSize: '12px',
                color: '#8b5cf6',
                padding: '4px 8px'
              }}
            >
              一键填充演示账户
            </Button>
          </div>
        </Space>
      </Card>
    </div>
  )
}

export default Login