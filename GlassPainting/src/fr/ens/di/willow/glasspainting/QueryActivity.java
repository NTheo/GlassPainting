package fr.ens.di.willow.glasspainting;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import fr.ens.di.willow.glasspainting.R;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import android.app.Activity;
import android.content.Context;
import android.media.AudioManager;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.WindowManager;

import com.google.android.glass.media.Sounds;
import com.google.android.glass.touchpad.Gesture;
import com.google.android.glass.touchpad.GestureDetector;

public class QueryActivity extends Activity implements CvCameraViewListener2
{

	private static final String TAG = "Activity";

	private Mat mRgba;
	private static final boolean bag = true;
	private static final int nbOfPaintings = 10;
	private Mat[] descriptors;
	private Mat[] mats;
	private long[] addrMats;
	private Mat index;
	private long[] addrDescriptors;
	private Mat voc;
	private long addrHist;
	private int transform;
	private GestureDetector mGestureDetector;
	private CameraBridgeViewBase mOpenCvCameraView;
	/**
	 * best response in last retrieval
	 */
	private int r;
	/**
	 * Descriptions of painting
	 * TODO put this as app resource
	 */
	private static final String[] infos =
		{"Corot",
		"David",
		"Ingres",
		"Benoist",
		"Delacroix",
		"painting 6",
		"painting 7",
		"painting 8",
		"painting 9",
		"painting 10",
		};
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this)
	{
		@Override
		public void onManagerConnected(int status)
		{
			switch (status)
			{
			case LoaderCallbackInterface.SUCCESS:
			{
				Log.i(TAG, "OpenCV loaded successfully");

				// Load native library after(!) OpenCV initialization
				System.loadLibrary("retrieval_demo");
				r = -1;
				if(!bag)
				{	
					addrDescriptors = new long[nbOfPaintings];
					descriptors = new Mat[nbOfPaintings];
					for (int i = 0; i < nbOfPaintings; i++)
					{
						descriptors[i] = Mat.zeros(500, 32, CvType.CV_8U);
						addrDescriptors[i] = descriptors[i].getNativeObjAddr();
					}
					load(addrDescriptors);
				}
				else
				{
					//hist = Mat.zeros(10000, 100, CvType.CV_32F);
					voc = Mat.zeros(500, 32, CvType.CV_8U);
					index = Mat.zeros(500, 2500, CvType.CV_8U);
					mats = new Mat[500];
					addrMats = new long[500];
					for(int i=0; i<500; i++)
					{
						mats[i] = Mat.zeros(2500, 32, CvType.CV_8U);
						addrMats[i] = mats[i].getNativeObjAddr();
					}
					addrHist = load2(voc.getNativeObjAddr(), addrMats, index.getNativeObjAddr());
				}
				mOpenCvCameraView.enableView();
			}
				break;
			default:
			{
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public QueryActivity()
	{
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState)
	{
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.face_detect_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
		mGestureDetector = createGestureDetector(this);
		transform = 0;
	}

	private GestureDetector createGestureDetector(Context context)
	{
		GestureDetector gestureDetector = new GestureDetector(context);
		// Create a base listener for generic gestures
		gestureDetector.setBaseListener(new GestureDetector.BaseListener()
		{
			@Override
			public boolean onGesture(Gesture gesture)
			{
				if (gesture == Gesture.TAP)
				{
		            AudioManager am = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
		            am.playSoundEffect(Sounds.TAP);
					Log.i(TAG, "Gesture TAP detected");
					transform = (transform + 1) % 3;
					return true;
				}
				return false;
			}
		});
		gestureDetector.setFingerListener(new GestureDetector.FingerListener()
		{
			@Override
			public void onFingerCountChanged(int previousCount, int currentCount)
			{
			}
		});
		gestureDetector.setScrollListener(new GestureDetector.ScrollListener()
		{
			@Override
			public boolean onScroll(float displacement, float delta,
					float velocity)
			{
				return false;
			}
		});
		return gestureDetector;
	}

	public boolean onGenericMotionEvent(MotionEvent event)
	{
		if (mGestureDetector != null)
		{
			return mGestureDetector.onMotionEvent(event);
		}
		return false;
	}

	@Override
	public void onPause()
	{
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume()
	{
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_8, this,
				mLoaderCallback);
	}

	public void onDestroy()
	{
		super.onDestroy();
		mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height)
	{
		mRgba = new Mat();
	}

	public void onCameraViewStopped()
	{
		mRgba.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame)
	{

		mRgba = inputFrame.rgba();

		switch (transform)
		{
		case 0:
			if(r==-1)
			{
				Core.rectangle(mRgba, new Point(35, 5), new Point(350, 60), new Scalar(255, 255, 255), Core.FILLED);
				Core.putText(mRgba, "Tap twice", new Point(40, 50),
					Core.FONT_HERSHEY_COMPLEX, 1.5, new Scalar(0, 0, 0));
			}
			else
			{
				Core.rectangle(mRgba, new Point(35, 220), new Point(550, 310), new Scalar(255, 255, 255), Core.FILLED);
				Core.putText(mRgba, infos[r], new Point(40, 300),
						Core.FONT_HERSHEY_COMPLEX, 1.5, new Scalar(0, 0, 0));
			}
			break;
		case 1:
			detect(mRgba.getNativeObjAddr());
			break;
		case 2:
            AudioManager am = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            if(!bag)
            	r = rank(mRgba.getNativeObjAddr(), addrDescriptors);
            if(bag)
            	r = rank2(mRgba.getNativeObjAddr(), voc.getNativeObjAddr(), addrMats, index.getNativeObjAddr(), addrHist);
            am.playSoundEffect(Sounds.SUCCESS);
			transform = 0;
			Log.i("answer", String.format("best answer is %d",r));
		}
		Log.i(TAG.concat("size"),
				String.format("%dx%d", mRgba.rows(), mRgba.cols()));
		return mRgba;
	}

	public static native void detect(long addrmRgba);

	public static native int rank(long addrmRgba, long[] addrDescriptors);
	public static native int rank2(long addrmRgba, long addrVoc, long[] addrMats, long addrIndex, long addrHist);
	

	public static native void load(long[] addrDescriptors);
	public static native long load2(long addrvoc, long[] addrMats, long addrIndex);
	
}
