import { useState, useRef } from 'react';
import apiClient from '../api/client';
import type { UploadResponse } from '../types';

interface Props {
  onUploaded: (data: UploadResponse) => void;
}

export default function ImageUploader({ onUploaded }: Props) {
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setLoading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const { data } = await apiClient.post<UploadResponse>('/upload', form);
      onUploaded(data);
    } catch {
      // Axios interceptor already logged the error
      alert('上传失败，请检查后端是否运行或图片格式是否正确');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="uploader">
      <input
        type="file"
        ref={fileRef}
        accept=".jpg,.jpeg,.png,.webp"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      <button
        onClick={() => fileRef.current?.click()}
        disabled={loading}
        className="upload-btn"
      >
        {loading ? '上传中...' : '选择图片'}
      </button>
    </div>
  );
}
