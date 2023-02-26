import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd

input_array = None


async def input_streamer():

    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()
    
    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
    
    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def process_audio_buffer():
    
    global input_array
    
    async for indata, status in input_streamer():

        indata_flattened = abs(indata.flatten())

        # discard buffers that contain mostly silence
        if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
            continue

        if (input_array is not None):
            input_array = np.concatenate((input_array, indata), dtype='int16')
        else:
            input_array = indata

        # concatenate buffers if the end of the current buffer is not silent
        if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
            continue
        else:
            local_ndarray = input_array.copy()
            input_array = None
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            result = model.transcribe(indata_transformed, language=LANGUAGE)
            print(result["text"])

        del local_ndarray
        del indata_flattened

        
async def run_asr_streaming():
    
    print('\nListening ...\n')
    
    audio_task = asyncio.create_task(process_audio_buffer())
    
    while True:
        await asyncio.sleep(1.0)
    audio_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nstream closed')

run_asr_streaming()
