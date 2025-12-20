import os
import sys
import json
import time
import pandas as pd 

# Đảm bảo Python nhìn thấy packages trong thư mục src
sys.path.append(os.getcwd())

from src.config import Config
from src.system_model import SystemModel
from src.ga_core import GACore
from src.ngsaii import NSGAII

def main():
    print("Bắt đầu hệ thống tối ưu hóa NSGA-II Fog Computing\n")

    # Cấu hình
    cfg = Config()
    
    cfg.number_generations = 100  # Sửa từ 20 thành 100
    
    # Đường dẫn file input (đảm bảo folder scenarios)
    cfg.topology_json = "scenarios/networkDefinition.json"
    cfg.application_json = "scenarios/appDefinition.json"

    # Kiểm tra file tồn tại
    if not os.path.exists(cfg.topology_json):
        print(f"Lỗi: Không tìm thấy {cfg.topology_json}")
        return

    # 2. Load Model
    print("📥 Đang tải System Model...")
    system = SystemModel(cfg)
    try:
        system.load()
        print(f"Đã tải Topology: {len(system.fog_nodes)} nodes")
    except Exception as e:
        print(f"Lỗi tải Model: {e}")
        return

    # 3. Khởi tạo GA
    # Lưu ý: Khi khởi tạo GACore, nó sẽ lấy population_size từ cfg (đã sửa thành 100 ở trên)
    core = GACore(system, cfg)
    nsga = NSGAII(core)

    # 4. Chạy tiến hóa
    print(f"🧬 Đang chạy {cfg.number_generations} thế hệ...")
    start_time = time.time()
    
    # Chạy hàm evolve nhưng KHÔNG dùng giá trị trả về (vì nó đã bị lọc)
    nsga.evolve()
    
    # --- SỬA ĐỔI QUAN TRỌNG ---
    # Thay vì lấy pareto_front, ta lấy trực tiếp toàn bộ quần thể từ core
    # Điều này đảm bảo bạn lấy đủ 100 cá thể cuối cùng (Rank 1, 2, 3...)
    final_pop = nsga.core.population_pt
    
    duration = time.time() - start_time
    print(f"✅ Hoàn thành sau {duration:.2f} giây.")

    # 5. Xuất kết quả ra màn hình & CSV
    print(f"\n🏆 Quần thể cuối cùng có {len(final_pop.population)} giải pháp.")
    
    results = []
    # Duyệt qua final_pop thay vì pareto_front
    for i, fit in enumerate(final_pop.fitness):
        # Kiểm tra nếu fitness tồn tại (để an toàn)
        if fit:
            results.append({
                "ID": i,
                "Latency": fit.get("latency"),
                "Spread": fit.get("spread"),
                "UnderUtil": fit.get("underutilization")
            })
    
    df = pd.DataFrame(results)
    print(df.head()) # In 5 dòng đầu
    print(f"Tổng số dòng dữ liệu: {len(df)}") # Kiểm tra lại số lượng
    
    # Lưu file CSV
    df.to_csv("ket_qua_pareto.csv", index=False)
    print("\n💾 Đã lưu kết quả chi tiết vào file: ket_qua_pareto.csv")

if __name__ == "__main__":
    main()