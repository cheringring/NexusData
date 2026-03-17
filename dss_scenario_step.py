"""
[DSS 시나리오 Python 스텝 코드]

이 코드를 Dataiku DSS 프로젝트(STREAMRIT_TEST_1)에
시나리오 "NexusData_Publish"의 Python 스텝으로 등록하세요.

동작:
1. Managed Folder "nexusdata_charts"에서 미게시 HTML 파일을 찾음
2. dataiku.insights.save_data()로 Static Insight 생성
3. 대시보드에 타일 자동 추가
4. 처리 완료된 파일의 JSON에 "published": true 마킹

시나리오 설정:
- 이름: NexusData_Publish
- 타입: step_based
- 트리거: API 호출 (외부 앱에서 트리거)
- 스텝 1: Python 스텝 (아래 코드)
"""
import dataiku
import dataiku.insights
import json

FOLDER_NAME = "nexusdata_charts"
DASHBOARD_NAME = "NexusData Dashboard"


def run():
    client = dataiku.api_client()
    project = client.get_default_project()
    folder = dataiku.Folder(FOLDER_NAME)

    # 미게시 파일 찾기 (_history 폴더 제외)
    items = folder.list_paths_in_partition()
    json_files = [p for p in items if p.endswith(".json") and not p.startswith("/_history/")]

    published_count = 0
    for json_path in json_files:
        try:
            # JSON 메타 읽기
            with folder.get_download_stream(json_path) as stream:
                meta = json.loads(stream.read().decode("utf-8"))

            # 이미 게시된 파일 스킵
            if meta.get("published"):
                continue

            insight_id = meta.get("insight_id", "")
            label = meta.get("title", meta.get("label", ""))
            html_path = json_path.replace(".json", ".html")

            # HTML 파일 읽기
            try:
                with folder.get_download_stream(html_path) as stream:
                    html_content = stream.read()
            except Exception:
                print(f"[Publish] HTML 파일 없음: {html_path}")
                continue

            # Static Insight 게시
            dataiku.insights.save_data(
                insight_id,
                html_content,
                content_type="text/html",
                label=label,
            )
            print(f"[Publish] Insight 게시 완료: {insight_id} ({label})")

            # 대시보드에 타일 추가
            try:
                _add_to_dashboard(project, insight_id, label)
            except Exception as e:
                print(f"[Publish] 대시보드 타일 추가 실패: {e}")

            # JSON에 published 마킹
            meta["published"] = True
            folder.upload_stream(
                json_path,
                json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            published_count += 1

        except Exception as e:
            print(f"[Publish] 처리 실패 ({json_path}): {e}")

    print(f"[Publish] 완료: {published_count}건 게시")


def _add_to_dashboard(project, insight_id, label):
    """대시보드에 Insight 타일 추가"""
    dashboard = None
    for d in project.list_dashboards():
        if "nexusdata" in d.name.lower():
            dashboard = d.to_dashboard()
            break

    if dashboard is None:
        dashboard = project.create_dashboard(DASHBOARD_NAME)

    settings = dashboard.get_settings()
    raw = settings.get_raw()
    pages = raw.get("pages", [])

    if not pages:
        pages.append({"title": "분석 차트", "grid": {"tiles": []}})
        raw["pages"] = pages

    page = pages[0]
    grid = page.setdefault("grid", {"tiles": []})
    tiles = grid.setdefault("tiles", [])

    # 중복 체크
    for t in tiles:
        if t.get("insightId") == insight_id:
            return

    # 타일 위치 계산
    max_row = 0
    for t in tiles:
        box = t.get("box", {})
        max_row = max(max_row, box.get("top", 0) + box.get("height", 4))

    tiles.append({
        "insightId": insight_id,
        "insightType": "static_file",
        "tileType": "INSIGHT",
        "title": label,
        "autoTitle": False,
        "box": {"left": 0, "top": max_row, "width": 12, "height": 25},
        "tileParams": {},
        "clickAction": "DO_NOTHING",
        "resizeImageMode": "FIT_SIZE",
        "displayMode": "INSIGHT",
    })
    settings.save()
    print(f"[Publish] 대시보드 타일 추가: {label}")


# 실행
run()
