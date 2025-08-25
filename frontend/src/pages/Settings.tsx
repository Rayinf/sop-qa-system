import { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  Button,
  Switch,
  Select,
  Slider,
  Typography,
  Space,
  Divider,
  message,
  Row,
  Col,
  Tabs,
  InputNumber,
  Upload,
  Avatar,
  Modal,
  List,
  Tag,
  Popconfirm
} from 'antd'
import {
  SettingOutlined,
  UserOutlined,
  BellOutlined,
  SecurityScanOutlined,
  DatabaseOutlined,
  ApiOutlined,
  UploadOutlined,
  EditOutlined,
  DeleteOutlined,
  PlusOutlined,
  SaveOutlined,
  ReloadOutlined
} from '@ant-design/icons'
import type { UploadProps } from 'antd'
import { User } from '../types/index'
import { SettingsApi, SystemSettings, NotificationSettings, SecuritySettings } from '../services/settingsApi'
import { ApiService } from '../services/api'
import { qaApi } from '../services/qaApi'

const { Title, Text } = Typography
const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

interface SettingsProps {
  user?: User
}



// Embedding设置相关类型
interface EmbeddingSettings {
  embedding_mode: string
  current_mode: string
  api_config?: {
    model_name: string
    base_url: string
    dimensions: number
  }
  local_config?: {
    model_name: string
    device: string
  }
}

const Settings = ({ user }: SettingsProps) => {
  const [loading, setLoading] = useState(false)
  const [systemSettings, setSystemSettings] = useState<SystemSettings | null>(null)
  const [notificationSettings, setNotificationSettings] = useState<NotificationSettings | null>(null)
  const [securitySettings, setSecuritySettings] = useState<SecuritySettings | null>(null)
  const [embeddingSettings, setEmbeddingSettings] = useState<EmbeddingSettings | null>(null)
  const [ipModalVisible, setIpModalVisible] = useState(false)
  const [newIp, setNewIp] = useState('')
  const [form] = Form.useForm()
  const [notificationForm] = Form.useForm()
  const [securityForm] = Form.useForm()
  // LLM 模型相关状态
  const [availableLLMModels, setAvailableLLMModels] = useState<string[]>([])
  const [currentLLMModel, setCurrentLLMModel] = useState<string>('')
  const [kimiFiles, setKimiFiles] = useState<any[]>([])
  const [kimiFileLoading, setKimiFileLoading] = useState(false)
  const [llmLoading, setLlmLoading] = useState(false)

  // 加载系统设置
  const loadSystemSettings = async (): Promise<void> => {
    try {
      setLoading(true)
      const settings = await SettingsApi.getSystemSettings()
      setSystemSettings(settings)
      form.setFieldsValue(settings)
    } catch (error) {
      console.error('加载系统设置失败:', error)
      message.error('加载系统设置失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载通知设置
  const loadNotificationSettings = async (): Promise<void> => {
    try {
      const settings = await SettingsApi.getNotificationSettings()
      setNotificationSettings(settings)
      notificationForm.setFieldsValue(settings)
    } catch (error) {
      console.error('加载通知设置失败:', error)
      message.error('加载通知设置失败')
    }
  }

  // 加载安全设置
  const loadSecuritySettings = async (): Promise<void> => {
    try {
      const settings = await SettingsApi.getSecuritySettings()
      setSecuritySettings(settings)
      securityForm.setFieldsValue(settings)
    } catch (error) {
      console.error('加载安全设置失败:', error)
      message.error('加载安全设置失败')
    }
  }

  // 加载embedding设置
  const loadEmbeddingSettings = async (): Promise<void> => {
    try {
      setLoading(true)
      const response = await ApiService.get<EmbeddingSettings>('/api/v1/settings/embedding')
      setEmbeddingSettings(response)
    } catch (error) {
      console.error('加载embedding设置失败:', error)
      message.error('加载embedding设置失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载LLM模型设置
  const loadLLMSettings = async (): Promise<void> => {
    try {
      setLlmLoading(true)
      const [models, current] = await Promise.all([
        qaApi.getAvailableModels(),
        qaApi.getCurrentModel()
      ])
      setAvailableLLMModels(models)
      setCurrentLLMModel(current)
    } catch (error) {
      console.error('加载LLM模型信息失败:', error)
      message.error('加载LLM模型信息失败')
    } finally {
      setLlmLoading(false)
    }
  }

  // 切换LLM模型
  const handleSwitchLLMModel = async (modelName: string): Promise<void> => {
    try {
      setLlmLoading(true)
      const res = await qaApi.switchModel(modelName)
      if (res?.success) {
        setCurrentLLMModel(modelName)
        message.success(`已切换到模型：${modelName}`)
        
        // 如果切换到Kimi模型，加载文件列表
        if (modelName.toLowerCase().includes('kimi')) {
          loadKimiFiles()
        }
      } else {
        message.error(res?.message || '切换模型失败')
      }
    } catch (error) {
      console.error('切换LLM模型失败:', error)
      message.error('切换LLM模型失败')
    } finally {
      setLlmLoading(false)
    }
  }

  // 加载Kimi文件列表
  const loadKimiFiles = async (): Promise<void> => {
    try {
      setKimiFileLoading(true)
      const result = await qaApi.getKimiFiles()
      if (result.success) {
        setKimiFiles(result.files)
      }
    } catch (error) {
      console.error('加载Kimi文件列表失败:', error)
    } finally {
      setKimiFileLoading(false)
    }
  }

  // 处理Kimi文件上传
  const handleKimiFileUpload = async (file: File): Promise<boolean> => {
    try {
      setKimiFileLoading(true)
      const result = await qaApi.uploadFileToKimi(file)
      if (result.success) {
        message.success(`文件 ${file.name} 上传成功`)
        loadKimiFiles() // 重新加载文件列表
      } else {
        message.error('文件上传失败')
      }
    } catch (error) {
      console.error('文件上传失败:', error)
      message.error('文件上传失败')
    } finally {
      setKimiFileLoading(false)
    }
    return false // 阻止默认上传行为
  }

  // 删除Kimi文件
  const handleDeleteKimiFile = async (fileId: string): Promise<void> => {
    try {
      setKimiFileLoading(true)
      const result = await qaApi.deleteKimiFile(fileId)
      if (result.success) {
        message.success('文件删除成功')
        loadKimiFiles() // 重新加载文件列表
      } else {
        message.error('文件删除失败')
      }
    } catch (error) {
      console.error('文件删除失败:', error)
      message.error('文件删除失败')
    } finally {
      setKimiFileLoading(false)
    }
  }

  useEffect(() => {
    loadSystemSettings()
    loadNotificationSettings()
    loadSecuritySettings()
    loadEmbeddingSettings()
    loadLLMSettings()
  }, [])

  // 切换embedding模式
  const handleSwitchEmbeddingMode = async (mode: string): Promise<void> => {
    try {
      setLoading(true)
      await ApiService.post('/api/v1/settings/embedding/switch', { mode })
      message.success(`embedding模式已切换到: ${mode === 'api' ? 'API模式' : '本地模式'}`)
      // 重新加载设置
      await loadEmbeddingSettings()
    } catch (error) {
      console.error('切换embedding模式失败:', error)
      message.error('切换embedding模式失败')
    } finally {
      setLoading(false)
    }
  }

  // 保存系统设置
  const handleSaveSystemSettings = async (): Promise<void> => {
    try {
      setLoading(true)
      const values = await form.validateFields()
      
      const updatedSettings = await SettingsApi.updateSystemSettings(values)
      setSystemSettings(updatedSettings)
      message.success('系统设置保存成功')
    } catch (error) {
      console.error('保存系统设置失败:', error)
      message.error('保存系统设置失败')
    } finally {
      setLoading(false)
    }
  }

  // 保存通知设置
  const handleSaveNotificationSettings = async (): Promise<void> => {
    try {
      setLoading(true)
      const values = await notificationForm.validateFields()
      
      const updatedSettings = await SettingsApi.updateNotificationSettings(values)
      setNotificationSettings(updatedSettings)
      message.success('通知设置保存成功')
    } catch (error) {
      console.error('保存通知设置失败:', error)
      message.error('保存通知设置失败')
    } finally {
      setLoading(false)
    }
  }

  // 保存安全设置
  const handleSaveSecuritySettings = async (): Promise<void> => {
    try {
      setLoading(true)
      const values = await securityForm.validateFields()
      
      const updatedSettings = await SettingsApi.updateSecuritySettings(values)
      setSecuritySettings(updatedSettings)
      message.success('安全设置保存成功')
    } catch (error) {
      console.error('保存安全设置失败:', error)
      message.error('保存安全设置失败')
    } finally {
      setLoading(false)
    }
  }

  // 添加IP地址
  const handleAddIp = (): void => {
    if (!newIp.trim()) {
      message.error('请输入有效的IP地址')
      return
    }
    
    const currentIps = securityForm.getFieldValue('ip_whitelist') || []
    const updatedIps = [...currentIps, newIp.trim()]
    
    securityForm.setFieldsValue({ ip_whitelist: updatedIps })
    setSecuritySettings((prev: SecuritySettings | null) => prev ? { ...prev, ip_whitelist: updatedIps } : null)
    setNewIp('')
    setIpModalVisible(false)
    message.success('IP地址添加成功')
  }

  // 删除IP地址
  const handleRemoveIp = (ipToRemove: string): void => {
    const currentIps = securityForm.getFieldValue('ip_whitelist') || []
    const updatedIps = currentIps.filter((ip: string) => ip !== ipToRemove)
    
    securityForm.setFieldsValue({ ip_whitelist: updatedIps })
    setSecuritySettings((prev: SecuritySettings | null) => prev ? { ...prev, ip_whitelist: updatedIps } : null)
    message.success('IP地址删除成功')
  }

  // 重置设置
  const handleResetSettings = (): void => {
    Modal.confirm({
      title: '确认重置',
      content: '确定要重置所有设置到默认值吗？此操作不可撤销。',
      okText: '确认',
      cancelText: '取消',
      onOk: async () => {
        try {
          setLoading(true)
          await SettingsApi.resetAllSettings()
          
          // 重新加载所有设置
          await Promise.all([
            loadSystemSettings(),
            loadNotificationSettings(),
            loadSecuritySettings()
          ])
          
          message.success('设置已重置为默认值')
        } catch (error) {
          console.error('重置设置失败:', error)
          message.error('重置设置失败')
        } finally {
          setLoading(false)
        }
      }
    })
  }

  // 文件上传配置
  const uploadProps: UploadProps = {
    name: 'file',
    action: '/api/upload/logo',
    headers: {
      authorization: 'Bearer ' + localStorage.getItem('access_token')
    },
    onChange(info: any): void {
      if (info.file.status === 'done') {
        message.success(`${info.file.name} 上传成功`)
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} 上传失败`)
      }
    }
  }

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>系统设置</Title>
        <Text type="secondary">
          配置系统参数、通知设置和安全策略。
        </Text>
      </div>

      <Tabs defaultActiveKey="system">
        {/* 系统设置 */}
        <TabPane tab={<span><SettingOutlined />系统设置</span>} key="system">
          <Card
            title="基本设置"
            extra={
              <Space>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={loadSystemSettings}
                  loading={loading}
                >
                  重新加载
                </Button>
                <Button
                  type="primary"
                  icon={<SaveOutlined />}
                  onClick={handleSaveSystemSettings}
                  loading={loading}
                >
                  保存设置
                </Button>
              </Space>
            }
          >
            <Form
              form={form}
              layout="vertical"
              initialValues={systemSettings || {}}
            >
              <Row gutter={[24, 0]}>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="site_name"
                    label="站点名称"
                    rules={[{ required: true, message: '请输入站点名称' }]}
                  >
                    <Input placeholder="请输入站点名称" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="default_user_role"
                    label="默认用户角色"
                    rules={[{ required: true, message: '请选择默认用户角色' }]}
                  >
                    <Select placeholder="选择默认角色">
                      <Option value="user">普通用户</Option>
                      <Option value="manager">管理员</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                name="site_description"
                label="站点描述"
              >
                <TextArea rows={3} placeholder="请输入站点描述" />
              </Form.Item>

              <Divider>文件上传设置</Divider>

              <Row gutter={[24, 0]}>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="max_file_size"
                    label="最大文件大小 (MB)"
                    rules={[{ required: true, message: '请设置最大文件大小' }]}
                  >
                    <InputNumber
                      min={1}
                      max={1000}
                      style={{ width: '100%' }}
                      placeholder="请输入最大文件大小"
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="max_questions_per_day"
                    label="每日最大提问数"
                    rules={[{ required: true, message: '请设置每日最大提问数' }]}
                  >
                    <InputNumber
                      min={1}
                      max={10000}
                      style={{ width: '100%' }}
                      placeholder="请输入每日最大提问数"
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                name="allowed_file_types"
                label="允许的文件类型"
                rules={[{ required: true, message: '请选择允许的文件类型' }]}
              >
                <Select
                  mode="multiple"
                  placeholder="选择允许的文件类型"
                  options={[
                    { label: 'PDF', value: 'pdf' },
                    { label: 'Word文档', value: 'doc' },
                    { label: 'Word文档(新)', value: 'docx' },
                    { label: '文本文件', value: 'txt' },
                    { label: 'Markdown', value: 'md' },
                    { label: 'Excel', value: 'xlsx' },
                    { label: 'PowerPoint', value: 'pptx' }
                  ]}
                />
              </Form.Item>

              <Divider>功能开关</Divider>

              <Row gutter={[24, 16]}>
                <Col xs={24} sm={12} md={8}>
                  <Form.Item
                    name="enable_registration"
                    label="允许用户注册"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Form.Item
                    name="enable_email_verification"
                    label="邮箱验证"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Form.Item
                    name="enable_file_upload"
                    label="文件上传"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Form.Item
                    name="enable_logging"
                    label="系统日志"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Form.Item
                    name="backup_enabled"
                    label="自动备份"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Form.Item
                    name="maintenance_mode"
                    label="维护模式"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>高级设置</Divider>

              <Row gutter={[24, 0]}>
                <Col xs={24} md={8}>
                  <Form.Item
                    name="session_timeout"
                    label="会话超时 (分钟)"
                  >
                    <InputNumber
                      min={5}
                      max={1440}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item
                    name="api_rate_limit"
                    label="API速率限制 (次/小时)"
                  >
                    <InputNumber
                      min={100}
                      max={10000}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item
                    name="log_level"
                    label="日志级别"
                  >
                    <Select>
                      <Option value="debug">调试</Option>
                      <Option value="info">信息</Option>
                      <Option value="warning">警告</Option>
                      <Option value="error">错误</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>

              <Divider>Embedding设置</Divider>

              <Row gutter={[24, 16]}>
                <Col xs={24} md={12}>
                  <Form.Item label="Embedding模式">
                    <Select
                      value={embeddingSettings?.current_mode || 'api'}
                      onChange={handleSwitchEmbeddingMode}
                      loading={loading}
                      style={{ width: '100%' }}
                    >
                      <Option value="api">
                        <Space>
                          <ApiOutlined />
                          API模式
                        </Space>
                      </Option>
                      <Option value="local">
                        <Space>
                          <DatabaseOutlined />
                          本地模式
                        </Space>
                      </Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="当前配置">
                    <div style={{ padding: '8px 12px', backgroundColor: '#f5f5f5', borderRadius: '6px' }}>
                      {embeddingSettings?.current_mode === 'api' ? (
                        <div>
                          <Text strong>API配置:</Text>
                          <br />
                          <Text type="secondary">
                            模型: {embeddingSettings?.api_config?.model_name || 'Qwen3'}
                          </Text>
                          <br />
                          <Text type="secondary">
                            维度: {embeddingSettings?.api_config?.dimensions || 1024}
                          </Text>
                        </div>
                      ) : (
                        <div>
                          <Text strong>本地配置:</Text>
                          <br />
                          <Text type="secondary">
                            模型: {embeddingSettings?.local_config?.model_name || 'sentence-transformers/all-MiniLM-L6-v2'}
                          </Text>
                          <br />
                          <Text type="secondary">
                            设备: {embeddingSettings?.local_config?.device || 'cpu'}
                          </Text>
                        </div>
                      )}
                    </div>
                  </Form.Item>
                </Col>
              </Row>

              <Divider>LLM模型设置</Divider>

              <Row gutter={[24, 16]}>
                <Col xs={24} md={12}>
                  <Form.Item label="选择LLM模型">
                    <Select
                      value={currentLLMModel || undefined}
                      placeholder="请选择模型"
                      onChange={handleSwitchLLMModel}
                      loading={llmLoading}
                      style={{ width: '100%' }}
                      showSearch
                      filterOption={(input, option) => (option?.children as unknown as string).toLowerCase().includes(input.toLowerCase())}
                    >
                      {availableLLMModels.map(model => (
                        <Option key={model} value={model}>{model}</Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="当前模型">
                    <div style={{ padding: '8px 12px', backgroundColor: '#f5f5f5', borderRadius: '6px' }}>
                      <Text strong>正在使用:</Text>
                      <br />
                      <Text type="secondary">{currentLLMModel || '未设置'}</Text>
                    </div>
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </Card>
        </TabPane>

        {/* 通知设置 */}
        <TabPane tab={<span><BellOutlined />通知设置</span>} key="notifications">
          <Card
            title="通知偏好"
            extra={
              <Button
                type="primary"
                icon={<SaveOutlined />}
                onClick={handleSaveNotificationSettings}
                loading={loading}
              >
                保存设置
              </Button>
            }
          >
            <Form
              form={notificationForm}
              layout="vertical"
              initialValues={notificationSettings || {}}
            >
              <Row gutter={[24, 16]}>
                <Col xs={24} sm={12}>
                  <Form.Item
                    name="email_notifications"
                    label="邮件通知"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                  <Text type="secondary">接收重要系统通知邮件</Text>
                </Col>
                <Col xs={24} sm={12}>
                  <Form.Item
                    name="browser_notifications"
                    label="浏览器通知"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                  <Text type="secondary">在浏览器中显示通知</Text>
                </Col>
                <Col xs={24} sm={12}>
                  <Form.Item
                    name="new_document_alerts"
                    label="新文档提醒"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                  <Text type="secondary">有新文档上传时通知</Text>
                </Col>
                <Col xs={24} sm={12}>
                  <Form.Item
                    name="system_alerts"
                    label="系统警报"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                  <Text type="secondary">系统异常时发送警报</Text>
                </Col>
                <Col xs={24} sm={12}>
                  <Form.Item
                    name="weekly_reports"
                    label="周报"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                  <Text type="secondary">每周发送使用统计报告</Text>
                </Col>
                <Col xs={24} sm={12}>
                  <Form.Item
                    name="notification_sound"
                    label="通知声音"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                  <Text type="secondary">播放通知提示音</Text>
                </Col>
              </Row>
            </Form>
          </Card>
        </TabPane>

        {/* 安全设置 */}
        <TabPane tab={<span><SecurityScanOutlined />安全设置</span>} key="security">
          <Card
            title="安全策略"
            extra={
              <Button
                type="primary"
                icon={<SaveOutlined />}
                onClick={handleSaveSecuritySettings}
                loading={loading}
              >
                保存设置
              </Button>
            }
          >
            <Form
              form={securityForm}
              layout="vertical"
              initialValues={securitySettings || {}}
            >
              <Divider>密码策略</Divider>
              
              <Row gutter={[24, 0]}>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="password_min_length"
                    label="最小密码长度"
                    rules={[{ required: true, message: '请设置最小密码长度' }]}
                  >
                    <Slider
                      min={6}
                      max={20}
                      marks={{
                        6: '6',
                        8: '8',
                        12: '12',
                        16: '16',
                        20: '20'
                      }}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="max_login_attempts"
                    label="最大登录尝试次数"
                    rules={[{ required: true, message: '请设置最大登录尝试次数' }]}
                  >
                    <InputNumber
                      min={3}
                      max={10}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={[24, 16]}>
                <Col xs={24} sm={12} md={6}>
                  <Form.Item
                    name="password_require_uppercase"
                    label="要求大写字母"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={6}>
                  <Form.Item
                    name="password_require_lowercase"
                    label="要求小写字母"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={6}>
                  <Form.Item
                    name="password_require_numbers"
                    label="要求数字"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} md={6}>
                  <Form.Item
                    name="password_require_symbols"
                    label="要求特殊字符"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>会话管理</Divider>
              
              <Row gutter={[24, 0]}>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="session_timeout_minutes"
                    label="会话超时时间 (分钟)"
                    rules={[{ required: true, message: '请设置会话超时时间' }]}
                  >
                    <InputNumber
                      min={5}
                      max={1440}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item
                    name="enable_two_factor"
                    label="启用双因素认证"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>IP白名单</Divider>
              
              <Form.Item label="允许访问的IP地址">
                <div style={{ marginBottom: 16 }}>
                  <Button
                    type="dashed"
                    icon={<PlusOutlined />}
                    onClick={() => setIpModalVisible(true)}
                  >
                    添加IP地址
                  </Button>
                </div>
                
                <List
                  dataSource={securitySettings?.ip_whitelist || []}
                  renderItem={(ip: string) => (
                    <List.Item
                      actions={[
                        <Popconfirm
                          key="delete"
                          title="确定要删除这个IP地址吗？"
                          onConfirm={() => handleRemoveIp(ip)}
                          okText="确定"
                          cancelText="取消"
                        >
                          <Button
                            type="link"
                            icon={<DeleteOutlined />}
                            danger
                            size="small"
                          >
                            删除
                          </Button>
                        </Popconfirm>
                      ]}
                    >
                      <Tag color="blue">{ip}</Tag>
                    </List.Item>
                  )}
                />
              </Form.Item>
            </Form>
          </Card>
        </TabPane>
      </Tabs>

      {/* 重置设置按钮 */}
      <Card style={{ marginTop: 24, textAlign: 'center' }}>
        <Space size="large">
          <Button
            danger
            onClick={handleResetSettings}
          >
            重置所有设置
          </Button>
          <Text type="secondary">
            重置后所有设置将恢复为默认值
          </Text>
        </Space>
      </Card>

      {/* 添加IP地址模态框 */}
      <Modal
        title="添加IP地址"
        open={ipModalVisible}
        onOk={handleAddIp}
        onCancel={() => {
          setIpModalVisible(false)
          setNewIp('')
        }}
        okText="添加"
        cancelText="取消"
      >
        <Input
          placeholder="请输入IP地址或CIDR (例如: 192.168.1.100 或 192.168.1.0/24)"
          value={newIp}
          onChange={(e: any) => setNewIp(e.target.value)}
          onPressEnter={handleAddIp}
        />
      </Modal>
    </div>
  )
}

export default Settings