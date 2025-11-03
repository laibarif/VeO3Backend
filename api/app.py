import os
import time
import uuid
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve generated videos folder
app.mount("/generated_videos", StaticFiles(directory="generated_videos"), name="generated_videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://www.trajectri.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoCreateRequest(BaseModel):
    user_id: str
    prompt: str
    title: Optional[str] = "Untitled"
    aspect_ratio: Optional[str] = "16:9"
    resolution: Optional[str] = "720p"
    negative_prompt: Optional[str] = None

class VideoGenerationService:
    def __init__(self):
        self.client = None
        self.db_conn = None
        self.init_services()
    
    def init_services(self):
        load_dotenv()
        
        # Google AI client - using the new genai client (same as your working script)
        api_key = os.getenv("VEO3_API_KEY")
        if not api_key:
            raise ValueError("VEO3_API_KEY not found")
        
        self.client = genai.Client(api_key=api_key)
        print("Google AI Veo client initialized successfully")
        
        # Database connection using your Neon DB URL
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found")
        
        self.db_conn = psycopg2.connect(database_url, sslmode='require')
        print("Neon database connection established")

    def get_or_create_internal_user(self, clerk_id: str):
        """Return the internal UUID corresponding to a Clerk ID."""
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE clerk_id = %s", (clerk_id,))
            user = cursor.fetchone()
            if user:
                return user['id']

            # Create new user
            new_user_id = str(uuid.uuid4())
            now = datetime.now()
            cursor.execute("""
                INSERT INTO users (id, clerk_id, name, image_url, video_credits)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (new_user_id, clerk_id, "Unknown Name", "", 3))  # 3 free credits
            self.db_conn.commit()
            return new_user_id

    def check_user_credits(self, clerk_user_id: str) -> bool:
        """Check if user has video credits available"""
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get internal user ID
                internal_user_id = self.get_or_create_internal_user(clerk_user_id)
                
                # Check subscription credits
                cursor.execute("""
                    SELECT video_credits_remaining 
                    FROM user_subscriptions 
                    WHERE user_id = %s AND status = 'active' AND video_credits_remaining > 0
                """, (internal_user_id,))
                subscription = cursor.fetchone()
                
                if subscription and subscription['video_credits_remaining'] > 0:
                    return True
                
                # Check free credits
                cursor.execute("""
                    SELECT video_credits 
                    FROM users 
                    WHERE id = %s AND video_credits > 0
                """, (internal_user_id,))
                user = cursor.fetchone()
                
                return user and user['video_credits'] > 0
                
        except Exception as e:
            print(f"Error checking user credits: {e}")
            return False

    def use_video_credit(self, clerk_user_id: str, video_id: str) -> bool:
        """Use one video credit for the user"""
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                internal_user_id = self.get_or_create_internal_user(clerk_user_id)
                
                # Try to use subscription credit first
                cursor.execute("""
                    UPDATE user_subscriptions 
                    SET video_credits_remaining = video_credits_remaining - 1,
                        updated_at = %s
                    WHERE user_id = %s AND status = 'active' AND video_credits_remaining > 0
                    RETURNING id
                """, (datetime.now(), internal_user_id))
                
                subscription_updated = cursor.fetchone()
                
                if subscription_updated:
                    # Record credit usage
                    cursor.execute("""
                        INSERT INTO video_credits_usage (user_id, video_id, credits_used)
                        VALUES (%s, %s, 1)
                    """, (internal_user_id, video_id))
                    self.db_conn.commit()
                    return True
                
                # Try to use free credit
                cursor.execute("""
                    UPDATE users 
                    SET video_credits = video_credits - 1,
                        updated_at = %s
                    WHERE id = %s AND video_credits > 0
                    RETURNING id
                """, (datetime.now(), internal_user_id))
                
                user_updated = cursor.fetchone()
                
                if user_updated:
                    cursor.execute("""
                        INSERT INTO video_credits_usage (user_id, video_id, credits_used)
                        VALUES (%s, %s, 1)
                    """, (internal_user_id, video_id))
                    self.db_conn.commit()
                    return True
                
                return False
                
        except Exception as e:
            self.db_conn.rollback()
            print(f"Error using video credit: {e}")
            return False

    def create_video_record(self, clerk_user_id: str, title: str = "Untitled"):
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                video_id = str(uuid.uuid4())
                now = datetime.now()

                # Get internal UUID for the Clerk user
                internal_user_id = self.get_or_create_internal_user(clerk_user_id)

                cursor.execute("""
                    INSERT INTO videos (
                        id, user_id, title, description, visibility, 
                        mux_status, duration, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                """, (
                    video_id,
                    internal_user_id,
                    title,
                    f"AI Generated Video for {clerk_user_id}",
                    'private',
                    'generating',
                    0,
                    now,
                    now
                ))

                video_record = cursor.fetchone()
                self.db_conn.commit()
                print(f"Stored video for Clerk user: {clerk_user_id}")
                return video_record

        except Exception as e:
            self.db_conn.rollback()
            raise e
    
    def update_video_after_generation(self, video_id: str, preview_url: str, duration: int = None):
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    UPDATE videos 
                    SET mux_status = %s, 
                        duration = %s,
                        updated_at = %s,
                        visibility = %s,
                        preview_url = %s
                    WHERE id = %s
                    RETURNING *
                """, ('ready', duration, datetime.now(), 'public', preview_url, video_id))
            
                updated_video = cursor.fetchone()
                self.db_conn.commit()
                return updated_video
            
        except Exception as e:
            self.db_conn.rollback()
            raise e

    def update_video_status(self, video_id: str, status: str, error_message: str = None):
        """Update video status in database"""
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # First check if error_message column exists
                if error_message:
                    try:
                        cursor.execute("""
                            UPDATE videos 
                            SET mux_status = %s, 
                                updated_at = %s,
                                error_message = %s
                            WHERE id = %s
                        """, (status, datetime.now(), error_message, video_id))
                    except psycopg2.Error:
                        # If error_message column doesn't exist, update without it
                        cursor.execute("""
                            UPDATE videos 
                            SET mux_status = %s, 
                                updated_at = %s
                            WHERE id = %s
                        """, (status, datetime.now(), video_id))
                else:
                    cursor.execute("""
                        UPDATE videos 
                        SET mux_status = %s, 
                            updated_at = %s
                        WHERE id = %s
                    """, (status, datetime.now(), video_id))
                
                self.db_conn.commit()
                
        except Exception as e:
            self.db_conn.rollback()
            raise e

    def generate_video(self, prompt: str, aspect_ratio: str = "16:9", resolution: str = "720p", negative_prompt: str = None):
        """Generate video using Google Veo AI - same as your working script"""
        try:
            config = types.GenerateVideosConfig(
                negative_prompt=negative_prompt if negative_prompt else None,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                person_generation="allow_all"
            )

            operation = self.client.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt=prompt,
                config=config
            )
            return operation
        except Exception as e:
            raise RuntimeError(f"Failed to start video generation: {e}")

    def poll_operation(self, operation, poll_interval: int = 5):
        """Poll the generation operation until completion - same as your working script"""
        start_time = time.time()
        print("Generating video...")

        while not operation.done:
            elapsed = int(time.time() - start_time)
            print(f"Still generating... ({elapsed}s elapsed)")
            time.sleep(poll_interval)
            try:
                operation = self.client.operations.get(operation)
            except Exception as e:
                raise RuntimeError(f"Error while polling operation: {e}")

        print("Video generation completed!")
        return operation

    def save_video_to_storage(self, operation, output_dir: str = "generated_videos"):
        """Save the generated video and return the file path - same as your working script"""
        try:
            if not operation.response or not operation.response.generated_videos:
                raise ValueError("No video generated in the response.")

            generated_video = operation.response.generated_videos[0]

            os.makedirs(output_dir, exist_ok=True)
            video_filename = f"video_{int(time.time())}.mp4"
            video_path = os.path.join(output_dir, video_filename)

            self.client.files.download(file=generated_video.video)
            generated_video.video.save(video_path)

            print(f"Video saved to: {video_path}")
            return video_path
        except Exception as e:
            raise RuntimeError(f"Failed to save video: {e}")

# Initialize the service
video_service = VideoGenerationService()

@app.post("/api/videos/create")
async def create_video(request: VideoCreateRequest, background_tasks: BackgroundTasks):
    """Endpoint to replace your TRPC create procedure"""
    try:
        # Check if user has credits
        if not video_service.check_user_credits(request.user_id):
            raise HTTPException(
                status_code=402, 
                detail="Insufficient video credits. Please upgrade your subscription."
            )
        
        # Step 1: Create video record in database
        video_record = video_service.create_video_record(request.user_id, request.title)
        video_id = video_record['id']
        
        # Use one credit
        if not video_service.use_video_credit(request.user_id, video_id):
            raise HTTPException(
                status_code=402, 
                detail="Failed to use video credit. Please try again."
            )
        
        # Step 2: Start background video generation
        background_tasks.add_task(
            generate_video_background,
            video_id,
            request.prompt,
            request.aspect_ratio,
            request.resolution,
            request.negative_prompt
        )
        
        return {
            "success": True,
            "video": dict(video_record),
            "message": "Video generation started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("Error in /api/videos/create:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def generate_video_background(video_id: str, prompt: str, aspect_ratio: str, resolution: str, negative_prompt: str):
    """Background task to generate video"""
    try:
        print(f"Starting video generation for {video_id}")
        
        # Update status to generating
        video_service.update_video_status(video_id, "generating")
        
        # Generate video using the same approach as your working script
        operation = video_service.generate_video(prompt, aspect_ratio, resolution, negative_prompt)
        operation = video_service.poll_operation(operation)
        video_path = video_service.save_video_to_storage(operation)
        
        # Estimate duration (you might want to extract this from the video file)
        duration = 8000  # 8 seconds default
        
        # Update video record with generation results
        # Convert local path to a public URL for frontend
        preview_filename = os.path.basename(video_path)
        
        preview_url = f"http://veo3backend-production.up.railway.app/generated_videos/{preview_filename}"
        # preview_url = f"http://localhost:8000/generated_videos/{preview_filename}"

        video_service.update_video_after_generation(video_id, preview_url=preview_url, duration=duration)

        print(f"Video generation completed for {video_id}")
        
    except Exception as e:
        print(f"Video generation failed for {video_id}: {str(e)}")
        # Update status to failed
        video_service.update_video_status(video_id, "failed", str(e))

@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: str):
    """Check video generation status"""
    try:
        with video_service.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM videos WHERE id = %s", (video_id,))
            video = cursor.fetchone()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            return {"video": dict(video)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Video Generation API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)