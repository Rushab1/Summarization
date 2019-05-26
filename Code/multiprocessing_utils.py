import multiprocessing as mp

#job_list is the list of objects to be splitted in batches of <len(job_list) / parallelism>
#processor_args = other args to be passed to processor
def multiprocessing_func(processor, job_list, parallelism, processor_args, writer=None, writer_args=[]):
    manager = mp.Manager()
    pool = mp.Pool()
    JobQueue = manager.Queue()
    jobs = []

    n = len(job_list)
    h = int(n/parallelism)

    if writer != None:
        args = list(writer_args)
        args.append(JobQueue)
        args = tuple (args)
        print(args)
        writer_job = pool.apply_async(writer, args)

    for i in range(0, len(job_list), h):
        args = [job_list[i:i+h]]
        args.extend(processor_args)
        args.append(JobQueue)

        args = tuple(args)

        job = pool.apply_async(processor, args)
        jobs.append(job)

    for job in jobs:
        job.get()
    JobQueue.put("kill")

    if writer != None:
        output = writer_job.get()

    pool.close()
    pool.join()

    return output
