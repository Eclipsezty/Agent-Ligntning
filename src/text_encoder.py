import pandas as pd


def _safe_get(row, col_name):
    """
    安全取值：如果是 NaN 或列不存在，就返回 None，避免出现 'nan' 这种字符串。
    """
    if col_name not in row.index:
        return None
    value = row[col_name]
    if pd.isna(value):
        return None
    return value


def flow_to_text(row: pd.Series) -> str:
    """
    将一行流量记录编码为一段给大模型看的英文描述文本。

    输入: pandas 的一行 (Series)，例如 df.iloc[0]
    输出: 一个字符串，用于喂给 LLM
    """
    parts = []

    # 1) 基本协议信息
    proto = _safe_get(row, "protocol_x")
    app = _safe_get(row, "application_name")
    app_cat = _safe_get(row, "application_category_name")

    basic = "This is a single network flow record."
    if proto:
        basic += f" The transport protocol is {proto}."
    if app:
        basic += f" The application is identified as {app}."
    if app_cat:
        basic += f" The application category is {app_cat}."
    parts.append(basic)

    # 2) 端口 & 通信方向
    sport = _safe_get(row, "sport")
    dport = _safe_get(row, "dport")
    is_common_port = _safe_get(row, "is_server_port_common")

    port_desc = []
    if sport is not None and dport is not None:
        port_desc.append(f"Source port is {int(sport)}, destination port is {int(dport)}.")
    if is_common_port is not None:
        if is_common_port == 1:
            port_desc.append("The destination port is a common server port.")
        elif is_common_port == 0:
            port_desc.append("The destination port is not a typical server port.")
    if port_desc:
        parts.append(" ".join(port_desc))

    # 3) 字节 & 包统计
    src_bytes = _safe_get(row, "src2dst_bytes")
    dst_bytes = _safe_get(row, "dst2src_bytes")
    src_pkts = _safe_get(row, "src2dst_packets")
    dst_pkts = _safe_get(row, "dst2src_packets")
    total_pkts = _safe_get(row, "bidirectional_packets")
    total_bytes = _safe_get(row, "bidirectional_bytes")

    traffic_desc = []
    if total_pkts is not None:
        traffic_desc.append(f"Total packets: {int(total_pkts)}.")
    if total_bytes is not None:
        traffic_desc.append(f"Total bytes: {int(total_bytes)}.")
    if src_pkts is not None and dst_pkts is not None:
        traffic_desc.append(f"Client-to-server packets: {int(src_pkts)}, server-to-client packets: {int(dst_pkts)}.")
    if src_bytes is not None and dst_bytes is not None:
        traffic_desc.append(f"Client-to-server bytes: {int(src_bytes)}, server-to-client bytes: {int(dst_bytes)}.")
    if traffic_desc:
        parts.append(" ".join(traffic_desc))

    # 4) 时间相关统计
    dur = _safe_get(row, "bidirectional_duration_ms")
    src_piat = _safe_get(row, "src2dst_mean_piat_ms")
    dst_piat = _safe_get(row, "dst2src_mean_piat_ms")

    time_desc = []
    if dur is not None:
        time_desc.append(f"The flow duration is about {round(float(dur), 2)} milliseconds.")
    if src_piat is not None and dst_piat is not None:
        time_desc.append(
            f"Average inter-arrival time from client to server is {round(float(src_piat), 2)} ms, "
            f"and from server to client is {round(float(dst_piat), 2)} ms."
        )
    if time_desc:
        parts.append(" ".join(time_desc))

    # 5) 上下行比例
    up_down = _safe_get(row, "up_down_ratio")
    down_up = _safe_get(row, "down_up_ratio")
    if up_down is not None and down_up is not None:
        parts.append(
            f"The up/down byte ratio is {round(float(up_down), 3)}, "
            f"and the down/up ratio is {round(float(down_up), 3)}."
        )

    # 6) 域名 & UA
    host = _safe_get(row, "requested_server_name")
    ua = _safe_get(row, "user_agent")
    ctype = _safe_get(row, "content_type")

    app_layer_desc = []
    if host:
        app_layer_desc.append(f"The requested server name (domain) is '{host}'.")
    if ua:
        app_layer_desc.append(f"The HTTP User-Agent string is: {ua}.")
    if ctype:
        app_layer_desc.append(f"The content type is: {ctype}.")
    if app_layer_desc:
        parts.append(" ".join(app_layer_desc))

    # 7) TLS 指纹（如果存在）
    ja3 = _safe_get(row, "ja3")
    ja4 = _safe_get(row, "ja4")
    if ja3 or ja4:
        fp_parts = []
        if ja3:
            fp_parts.append(f"JA3 fingerprint: {ja3}.")
        if ja4:
            fp_parts.append(f"JA4 fingerprint: {ja4}.")
        parts.append(" ".join(fp_parts))

    # 8) 负载大小
    payload_size = _safe_get(row, "bytes_payload_size")
    if payload_size is not None:
        parts.append(f"The total payload byte size is {int(payload_size)}.")

    # 9) 总结提示（给大模型的任务指令可以在真正 prompt 里额外加）
    # 这里 encoder 只负责“事实描述”，不负责问问题

    return " ".join(parts)
