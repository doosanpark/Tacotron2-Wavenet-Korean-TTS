#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
학습 진행 상황을 실시간으로 모니터링하는 스크립트
"""
import os
import time
import glob
from datetime import datetime

def get_latest_log_dir():
    """가장 최근 로그 디렉토리 찾기"""
    log_base = 'logdir-tacotron2'
    dirs = glob.glob(os.path.join(log_base, 'moon+son_*'))
    if not dirs:
        return None
    # 최신 디렉토리 반환
    latest = max(dirs, key=os.path.getmtime)
    return latest

def monitor_training():
    """학습 로그 모니터링"""
    print("=" * 80)
    print("학습 진행 상황 모니터링 시작...")
    print("=" * 80)
    
    log_dir = get_latest_log_dir()
    if log_dir:
        log_file = os.path.join(log_dir, 'train.log')
        print(f"\n모니터링 중인 로그: {log_file}\n")
        
        # 파일 크기 추적
        last_size = 0
        if os.path.exists(log_file):
            last_size = os.path.getsize(log_file)
        
        while True:
            try:
                if os.path.exists(log_file):
                    current_size = os.path.getsize(log_file)
                    
                    # 새 내용이 추가되었는지 확인
                    if current_size > last_size:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            # 마지막 위치로 이동
                            f.seek(last_size)
                            new_content = f.read()
                            
                            if new_content.strip():
                                # Step 관련 로그만 필터링하여 출력
                                lines = new_content.split('\n')
                                for line in lines:
                                    if 'Step' in line or 'loss=' in line or 'Saving checkpoint' in line or 'Writing summary' in line:
                                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                                    elif 'Exiting due to exception' in line or 'Error' in line or 'failed' in line:
                                        print(f"\n[오류 발생!] {line}")
                                        
                        last_size = current_size
                
                time.sleep(2)  # 2초마다 확인
                
            except KeyboardInterrupt:
                print("\n모니터링 중단됨.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                time.sleep(5)
    else:
        print("로그 디렉토리를 찾을 수 없습니다. 학습이 시작되기를 기다리는 중...")
        time.sleep(5)

if __name__ == '__main__':
    monitor_training()

