// 跨页面共享的基础数据类型；字段与后端响应保持对应，UUID 用字符串传输，日期在展示时再解析。
export type Repository = {
  id: string;
  owner: string;
  name: string;
  full_name: string;
  provider?: string;
  api_base_url?: string;
  clone_url?: string | null;
  local_path?: string | null;
  checkout_mode?: string;
  description?: string;
  default_branch?: string;
  last_sync_at?: string | null;
};
