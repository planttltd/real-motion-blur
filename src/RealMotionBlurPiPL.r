#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "AE_General.r"

resource 'PiPL' (16000) {
	{
		Kind { AEEffect },
		Name { "Real Motion Blur" },
		Category { "Blur & Sharpen" },
		CodeMacARM64 { "EffectMain" },
		AE_PiPL_Version { 2, 0 },
		AE_Effect_Spec_Version { PF_PLUG_IN_VERSION, PF_PLUG_IN_SUBVERS },
		AE_Effect_Version { 524289 /* PF_VERSION(1,0,0,DEVELOP,1) */ },
		AE_Effect_Info_Flags { 0 },
		/* DEEP_COLOR_AWARE | CUSTOM_UI | WIDE_TIME_INPUT */
		AE_Effect_Global_OutFlags { 0x02008002 },
		/* SMART_RENDER | FLOAT_COLOR_AWARE | AUTOMATIC_WIDE_TIME_INPUT | THREADED_RENDERING (no GPU frames: see
		   kAEGPUFrames in RealMotionBlur.mm; with them this is 0x0A021400) */
		AE_Effect_Global_OutFlags_2 { 0x08021400 },
		AE_Effect_Match_Name { "RPLT RealMotionBlur" },
		AE_Reserved_Info { 0 },
		AE_Effect_Support_URL { "https://github.com/planttltd/real-motion-blur" }
	}
};

resource 'PiPL' (16001) {
	{
		Kind { AEEffect },
		Name { "Real Motion Blur Vectors" },
		Category { "Blur & Sharpen" },
		CodeMacARM64 { "VectorsMain" },
		AE_PiPL_Version { 2, 0 },
		AE_Effect_Spec_Version { PF_PLUG_IN_VERSION, PF_PLUG_IN_SUBVERS },
		AE_Effect_Version { 524289 /* PF_VERSION(1,0,0,DEVELOP,1) */ },
		AE_Effect_Info_Flags { 0 },
		/* DEEP_COLOR_AWARE | CUSTOM_UI */
		AE_Effect_Global_OutFlags { 0x02008000 },
		/* SMART_RENDER | FLOAT_COLOR_AWARE | THREADED_RENDERING (no GPU frames; with them 0x0A001400) */
		AE_Effect_Global_OutFlags_2 { 0x08001400 },
		AE_Effect_Match_Name { "RPLT RealMotionBlurVectors" },
		AE_Reserved_Info { 0 },
		AE_Effect_Support_URL { "https://github.com/planttltd/real-motion-blur" }
	}
};
