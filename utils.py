import datetime

log_info_file = None

def log_info(*args):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{dtstr}]", *args)
    if log_info_file:
        print(f"[{dtstr}]", *args, file=log_info_file, flush=True)
