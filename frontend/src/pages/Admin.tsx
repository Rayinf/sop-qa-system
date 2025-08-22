import { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Typography,
  Modal,
  Form,
  Input,
  Select,
  message,
  Popconfirm,
  Row,
  Col,
  Statistic,
  Tabs,
  List,
  Progress,
  Switch,
  DatePicker,
  Divider
} from 'antd'
import {
  UserOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined,
  SettingOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  QuestionCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons'
import type { TableColumnsType } from 'antd'
import { User } from '../types/index'
import { AdminApi, AdminUser, SystemStats, SystemLog } from '../services/adminApi'

const { Title, Text } = Typography
const { Option } = Select
const { TabPane } = Tabs
const { RangePicker } = DatePicker

interface AdminProps {
  user?: User
}

const Admin: React.FC = () => {
  const [users, setUsers] = useState<AdminUser[]>([])
  const [loading, setLoading] = useState(false)
  const [userModalVisible, setUserModalVisible] = useState(false)
  const [editingUser, setEditingUser] = useState<AdminUser | null>(null)
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [systemLogs, setSystemLogs] = useState<SystemLog[]>([])
  const [form] = Form.useForm()

  // 加载用户列表
  const loadUsers = async () => {
    setLoading(true)
    try {
      const usersData = await AdminApi.getUsers()
      setUsers(usersData)
    } catch (error) {
      console.error('加载用户数据失败:', error)
      message.error('加载用户数据失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载系统统计
  const loadSystemStats = async (): Promise<void> => {
    try {
      const statsData = await AdminApi.getSystemStats()
      setSystemStats(statsData)
    } catch (error) {
      console.error('加载系统统计失败:', error)
      message.error('加载系统统计失败')
    }
  }

  // 加载系统日志
  const loadSystemLogs = async (): Promise<void> => {
    try {
      const logsData = await AdminApi.getSystemLogs(50)
      setSystemLogs(logsData)
    } catch (error) {
      console.error('加载系统日志失败:', error)
      message.error('加载系统日志失败')
    }
  }

  useEffect(() => {
    loadUsers()
    loadSystemStats()
    loadSystemLogs()
  }, [])

  // 创建/编辑用户
  const handleSaveUser = async (): Promise<void> => {
    try {
      setLoading(true)
      const values = await form.validateFields()
      
      if (editingUser) {
        // 编辑用户
        const updatedUser = await AdminApi.updateUser(editingUser.id, values)
        setUsers((prev: AdminUser[]) => 
          prev.map((u: AdminUser) => 
            u.id === editingUser.id ? updatedUser : u
          )
        )
        message.success('用户信息更新成功')
      } else {
        // 创建新用户
        const newUser = await AdminApi.createUser(values)
        setUsers((prev: AdminUser[]) => [...prev, newUser])
        message.success('用户创建成功')
      }
      
      setUserModalVisible(false)
      setEditingUser(null)
      form.resetFields()
    } catch (error) {
      console.error('用户操作失败:', error)
      message.error('操作失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  // 删除用户
  const handleDeleteUser = async (userId: number): Promise<void> => {
    try {
      await AdminApi.deleteUser(userId)
      setUsers((prev: AdminUser[]) => prev.filter((u: AdminUser) => u.id !== userId))
      message.success('用户删除成功')
    } catch (error) {
      console.error('删除用户失败:', error)
      message.error('删除失败')
    }
  }

  // 切换用户状态
  const handleToggleUserStatus = async (userId: number, isActive: boolean): Promise<void> => {
    try {
      const updatedUser = await AdminApi.updateUser(userId, { is_active: isActive })
      setUsers((prev: AdminUser[]) => 
        prev.map((u: AdminUser) => 
          u.id === userId ? updatedUser : u
        )
      )
      message.success(`用户已${isActive ? '启用' : '禁用'}`)
    } catch (error) {
      console.error('切换用户状态失败:', error)
      message.error('操作失败')
    }
  }

  // 编辑用户
  const handleEditUser = (user: AdminUser): void => {
    setEditingUser(user)
    form.setFieldsValue({
      username: user.username,
      email: user.email,
      role: user.role,
      is_active: user.is_active
    })
    setUserModalVisible(true)
  }

  // 格式化时间
  const formatTime = (dateString: string | null): string => {
    if (!dateString) return '从未登录'
    return new Date(dateString).toLocaleString('zh-CN')
  }

  // 获取日志级别标签
  const getLogLevelTag = (level: string): JSX.Element => {
    const levelMap = {
      info: { color: 'blue', icon: <CheckCircleOutlined /> },
      warning: { color: 'orange', icon: <WarningOutlined /> },
      error: { color: 'red', icon: <CloseCircleOutlined /> },
      debug: { color: 'default', icon: <ClockCircleOutlined /> }
    }
    const config = levelMap[level as keyof typeof levelMap] || levelMap.debug
    return (
      <Tag color={config.color} icon={config.icon}>
        {level.toUpperCase()}
      </Tag>
    )
  }

  // 用户表格列定义
  const userColumns: TableColumnsType<AdminUser> = [
    {
      title: '用户名',
      dataIndex: 'username',
      key: 'username'
    },
    {
      title: '姓名',
      dataIndex: 'full_name',
      key: 'full_name'
    },
    {
      title: '邮箱',
      dataIndex: 'email',
      key: 'email'
    },
    {
      title: '角色',
      dataIndex: 'role',
      key: 'role',
      render: (role: string) => (
        <Tag color={role === 'admin' ? 'red' : 'blue'}>
          {role === 'admin' ? '管理员' : '用户'}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      render: (isActive: boolean, record: AdminUser) => (
        <Switch
          checked={isActive}
          onChange={(checked: boolean) => handleToggleUserStatus(record.id, checked)}
          checkedChildren="启用"
          unCheckedChildren="禁用"
        />
      )
    },
    {
      title: '最后登录',
      dataIndex: 'last_login',
      key: 'last_login',
      render: (time: string | null) => formatTime(time)
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => formatTime(time)
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: AdminUser) => (
        <Space>
          <Button
            type="link"
            icon={<EditOutlined />}
            onClick={() => handleEditUser(record)}
          >
            编辑
          </Button>
          <Popconfirm
            title="确定要删除这个用户吗？"
            onConfirm={() => handleDeleteUser(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Button type="link" icon={<DeleteOutlined />} danger>
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ]

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>系统管理</Title>
        <Text type="secondary">
          管理系统用户、监控系统状态和查看系统日志。
        </Text>
      </div>

      <Tabs defaultActiveKey="overview">
        {/* 系统概览 */}
        <TabPane tab="系统概览" key="overview">
          {/* 统计卡片 */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={24} sm={6}>
              <Card>
                <Statistic
                  title="总用户数"
                  value={systemStats?.total_users || 0}
                  prefix={<UserOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={6}>
              <Card>
                <Statistic
                  title="活跃用户"
                  value={systemStats?.active_users || 0}
                  prefix={<UserOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={6}>
              <Card>
                <Statistic
                  title="文档总数"
                  value={systemStats?.total_documents || 0}
                  prefix={<FileTextOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={6}>
              <Card>
                <Statistic
                  title="问答总数"
                  value={systemStats?.total_questions || 0}
                  prefix={<QuestionCircleOutlined />}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            {/* 系统健康状态 */}
            <Col xs={24} lg={12}>
              <Card title="系统健康状态">
                {systemStats?.system_health ? (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <Text>CPU 使用率</Text>
                        <Text>{systemStats.system_health.cpu_usage}%</Text>
                      </div>
                      <Progress 
                        percent={systemStats.system_health.cpu_usage} 
                        status={systemStats.system_health.cpu_usage > 80 ? 'exception' : 'normal'}
                      />
                    </div>
                    
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <Text>内存使用率</Text>
                        <Text>{systemStats.system_health.memory_usage}%</Text>
                      </div>
                      <Progress 
                        percent={systemStats.system_health.memory_usage}
                        status={systemStats.system_health.memory_usage > 85 ? 'exception' : 'normal'}
                        strokeColor={systemStats.system_health.memory_usage > 85 ? '#ff4d4f' : '#1890ff'}
                      />
                    </div>
                    
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <Text>磁盘使用率</Text>
                        <Text>{systemStats.system_health.disk_usage}%</Text>
                      </div>
                      <Progress 
                        percent={systemStats.system_health.disk_usage} 
                        status={systemStats.system_health.disk_usage > 90 ? 'exception' : 'normal'}
                      />
                    </div>
                    
                    <Divider style={{ margin: '12px 0' }} />
                    
                    <Row gutter={16}>
                      <Col span={8}>
                        <div>
                          <Text>数据库状态</Text>
                          <br />
                          <Tag color={systemStats.system_health.database_status === 'connected' ? 'green' : 'red'}>
                            {systemStats.system_health.database_status === 'connected' ? '正常' : '异常'}
                          </Tag>
                        </div>
                      </Col>
                      <Col span={8}>
                        <div>
                          <Text>Redis状态</Text>
                          <br />
                          <Tag color={systemStats.system_health.redis_status === 'connected' ? 'green' : 'red'}>
                            {systemStats.system_health.redis_status === 'connected' ? '正常' : '异常'}
                          </Tag>
                        </div>
                      </Col>
                      <Col span={8}>
                        <div>
                          <Text>API响应时间</Text>
                          <br />
                          <Text strong>{systemStats.system_health.api_response_time || 0}ms</Text>
                        </div>
                      </Col>
                    </Row>
                  </Space>
                ) : (
                  <Text type="secondary">加载中...</Text>
                )}
              </Card>
            </Col>

            {/* 今日活动 */}
            <Col xs={24} lg={12}>
              <Card title="今日活动">
                {systemStats?.recent_activity ? (
                  <Row gutter={[0, 16]}>
                    <Col span={24}>
                      <Statistic
                        title="新增用户"
                        value={0}
                        prefix={<UserOutlined />}
                      />
                    </Col>
                    <Col span={24}>
                      <Statistic
                        title="提问数量"
                        value={0}
                        prefix={<QuestionCircleOutlined />}
                      />
                    </Col>
                    <Col span={24}>
                      <Statistic
                        title="文档上传"
                        value={0}
                        prefix={<FileTextOutlined />}
                      />
                    </Col>
                    <Col span={24}>
                      <Statistic
                        title="系统错误"
                        value={0}
                        prefix={<WarningOutlined />}
                        valueStyle={{ 
                          color: '#52c41a'
                        }}
                      />
                    </Col>
                  </Row>
                ) : (
                  <Text type="secondary">加载中...</Text>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        {/* 用户管理 */}
        <TabPane tab="用户管理" key="users">
          <Card
            title="用户列表"
            extra={
              <Space>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={loadUsers}
                  loading={loading}
                >
                  刷新
                </Button>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => {
                    setEditingUser(null)
                    form.resetFields()
                    setUserModalVisible(true)
                  }}
                >
                  添加用户
                </Button>
              </Space>
            }
          >
            <Table
              columns={userColumns}
              dataSource={users}
              rowKey="id"
              loading={loading}
              pagination={{
                total: users.length,
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
              }}
            />
          </Card>
        </TabPane>

        {/* 系统日志 */}
        <TabPane tab="系统日志" key="logs">
          <Card
            title="系统日志"
            extra={
              <Space>
                <RangePicker />
                <Select defaultValue="all" style={{ width: 120 }}>
                  <Option value="all">全部级别</Option>
                  <Option value="info">信息</Option>
                  <Option value="warning">警告</Option>
                  <Option value="error">错误</Option>
                </Select>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={loadSystemLogs}
                >
                  刷新
                </Button>
              </Space>
            }
          >
            <List
              dataSource={systemLogs}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    title={
                      <Space>
                        {getLogLevelTag(item.level)}
                        <Text>{item.message}</Text>
                      </Space>
                    }
                    description={
                      <Space>
                        <Text type="secondary">{formatTime(item.timestamp)}</Text>
                        <Tag>{item.module}</Tag>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 用户编辑模态框 */}
      <Modal
        title={editingUser ? '编辑用户' : '添加用户'}
        open={userModalVisible}
        onOk={handleSaveUser}
        onCancel={() => {
          setUserModalVisible(false)
          setEditingUser(null)
          form.resetFields()
        }}
        confirmLoading={loading}
        okText="保存"
        cancelText="取消"
      >
        <Form
          form={form}
          layout="vertical"
        >
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input placeholder="请输入用户名" disabled={!!editingUser} />
          </Form.Item>
          
          <Form.Item
            name="email"
            label="邮箱"
            rules={[
              { required: true, message: '请输入邮箱' },
              { type: 'email', message: '请输入有效的邮箱地址' }
            ]}
          >
            <Input placeholder="请输入邮箱" />
          </Form.Item>
          

          
          <Form.Item
            name="role"
            label="角色"
            rules={[{ required: true, message: '请选择角色' }]}
          >
            <Select placeholder="选择角色">
              <Option value="user">用户</Option>
              <Option value="admin">管理员</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="is_active"
            label="状态"
            valuePropName="checked"
          >
            <Switch checkedChildren="启用" unCheckedChildren="禁用" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default Admin