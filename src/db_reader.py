# db_reader.py
import sqlite3
from config import DB_PATH

def get_all_descriptions_for_video(video_stem):
    """
    Returns list of tuples:
      (video_name, frame_index, frame_path, description, clip_id)

    video_stem should match the start of video_name, e.g.:
      video_stem = "classroom"
      video_name = "classroom_subclip_1"
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # classroom -> classroom_subclip_%
    like_pattern = f"{video_stem}_subclip_%"

    cur.execute("""
        SELECT d.video_name,
               d.frame_index,
               k.frame_path,
               d.description,
               d.clip_id
        FROM descriptions AS d
        LEFT JOIN keyframes AS k
          ON d.video_name = k.video_name
         AND d.frame_index = k.frame_index
        WHERE d.video_name LIKE ?
        ORDER BY d.id
    """, (like_pattern,))

    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_videos():
    """
    Return distinct video stems from video_name prefix before '_subclip_'.
    Example: 'classroom_subclip_1' -> 'classroom'
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT video_name FROM descriptions")
    video_names = [r[0] for r in cur.fetchall()]
    stems = set()
    for name in video_names:
        if "_subclip_" in name:
            stems.add(name.split("_subclip_")[0])
    conn.close()
    return list(stems)
