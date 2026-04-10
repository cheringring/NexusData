# -*- coding: utf-8 -*-
"""사용자별 채팅 히스토리 관리"""
import io
import os
import re
import json
from datetime import datetime
from typing import List


class HistoryManager:
    """
    사용자별 채팅 히스토리를 Managed Folder(Dataiku) 또는 로컬 JSON으로 저장/복원.
    저장 구조: _history/{user_id}/{chat_id}.json
    각 대화는 고유 chat_id를 가지며, 여러 대화를 독립적으로 관리.
    """
    FOLDER_NAME = "nexusdata_charts"
    HISTORY_PREFIX = "_history"

    def __init__(self, storage_dir: str = ".chat_history"):
        self.storage_dir = storage_dir
        self._in_dataiku = False
        self._folder = None
        try:
            import dataiku
            self._folder = dataiku.Folder(self.FOLDER_NAME)
            self._in_dataiku = True
        except Exception:
            os.makedirs(storage_dir, exist_ok=True)

    @staticmethod
    def get_user_id() -> str:
        """Dataiku 사용자 ID 가져오기 (없으면 'default')"""
        try:
            import dataiku
            client = dataiku.api_client()
            auth_info = client.get_auth_info()
            return auth_info.get("authIdentifier", "default")
        except Exception:
            return "default"

    @staticmethod
    def new_chat_id() -> str:
        """새 대화 ID 생성"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _chat_path(self, user_id: str, chat_id: str) -> str:
        if self._in_dataiku:
            return f"{self.HISTORY_PREFIX}/{user_id}/{chat_id}.json"
        return os.path.join(self.storage_dir, user_id, f"{chat_id}.json")

    def _folder_path(self, user_id: str, dataset_name: str) -> str:
        safe_ds_name = re.sub(r'[^\w\-]', '_', dataset_name)
        return f"{self.HISTORY_PREFIX}/{user_id}_{safe_ds_name}.json"

    def get_history_file(self, user_id: str, dataset_name: str) -> str:
        safe_ds_name = re.sub(r'[^\w\-]', '_', dataset_name)
        return os.path.join(self.storage_dir, f"{user_id}_{safe_ds_name}.json")

    def save_history(self, user_id: str, chat_id: str, messages: List[dict],
                     title: str = None, dataset_name: str = "") -> bool:
        """히스토리 저장"""
        try:
            if title is None:
                first_msg = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
                title = (first_msg[:25] + '...') if len(first_msg) > 25 else first_msg
                if not title:
                    title = dataset_name or chat_id

            data = {
                "chat_id": chat_id,
                "user_id": user_id,
                "dataset": dataset_name,
                "title": title,
                "last_updated": datetime.now().isoformat(),
                "messages": messages,
            }
            payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

            if self._in_dataiku and self._folder:
                self._folder.upload_stream(self._chat_path(user_id, chat_id), io.BytesIO(payload))
            else:
                dirpath = os.path.join(self.storage_dir, user_id)
                os.makedirs(dirpath, exist_ok=True)
                with open(self._chat_path(user_id, chat_id), 'w', encoding='utf-8') as f:
                    f.write(payload.decode("utf-8"))
            return True
        except Exception as e:
            print(f"히스토리 저장 실패: {e}")
            return False

    def load_history(self, user_id: str, chat_id: str) -> dict:
        """chat_id로 대화 전체 데이터 로드. 반환: {chat_id, dataset, title, messages}"""
        try:
            if self._in_dataiku and self._folder:
                path = self._chat_path(user_id, chat_id)
                with self._folder.get_download_stream(path) as f:
                    return json.loads(f.read().decode("utf-8"))
            else:
                path = self._chat_path(user_id, chat_id)
                if not os.path.exists(path):
                    return {}
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"히스토리 로드 실패: {e}")
            return {}

    def list_user_histories(self, user_id: str) -> List[dict]:
        """사용자의 모든 대화 목록 (최신순)"""
        histories = []
        try:
            if self._in_dataiku and self._folder:
                prefix = f"{self.HISTORY_PREFIX}/{user_id}/"
                for path in self._folder.list_paths_in_partition():
                    if path.startswith(prefix) and path.endswith(".json"):
                        try:
                            with self._folder.get_download_stream(path) as f:
                                data = json.loads(f.read().decode("utf-8"))
                                histories.append({
                                    "chat_id": data.get("chat_id", ""),
                                    "dataset": data.get("dataset", ""),
                                    "title": data.get("title", ""),
                                    "last_updated": data.get("last_updated", ""),
                                    "message_count": len(data.get("messages", [])),
                                })
                        except Exception:
                            continue
            else:
                dirpath = os.path.join(self.storage_dir, user_id)
                if os.path.isdir(dirpath):
                    for fname in os.listdir(dirpath):
                        if fname.endswith(".json"):
                            fpath = os.path.join(dirpath, fname)
                            try:
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    histories.append({
                                        "chat_id": data.get("chat_id", fname[:-5]),
                                        "dataset": data.get("dataset", ""),
                                        "title": data.get("title", ""),
                                        "last_updated": data.get("last_updated", ""),
                                        "message_count": len(data.get("messages", [])),
                                    })
                            except Exception:
                                continue
        except Exception:
            pass
        return sorted(histories, key=lambda x: x.get("last_updated", ""), reverse=True)

    def delete_history(self, user_id: str, chat_id: str) -> bool:
        """히스토리 삭제"""
        try:
            if self._in_dataiku and self._folder:
                self._folder.delete_path(self._chat_path(user_id, chat_id))
            else:
                path = self._chat_path(user_id, chat_id)
                if os.path.exists(path):
                    os.remove(path)
            return True
        except Exception:
            return False
