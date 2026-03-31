#Changelog:

#2026/03/19
- project big rework
- deleted list of extensions
- written own flux backend realization with normal mode and tiling generation (with UI control, including tiles setup)
- written own flux teacache/first block cache extension
- reworked flux kontext extension
- deleted list of files (unneeded)
- implemented consistent unload/load model logic with memory safe cleanup and load
- integrated repositories
- implemented local cache folder for almost all cache, including pip, uv (while installing), transformers etc.
- reworked ReActorZero code, now less weight, pre-configured
- enabled back "save in folders" setting
- cleaned config_ui from trash settings
- small fixes

#2026/01/24
- Florence-2 code moved to LeeAeron HugginFace repo

#2026/01/21
DUE TO MY FAULT, OLD MAIN BRANCH NOW IS 'DEAD', YOU HAE TO RE-DOWNLOAD AND RE-INSTALL CU121 BUILD TO GET NEW UPDATES.
- added ENTRY (10) into main menu: option to download ClarityHD Upscaler base model and LoRas.

#2026/01/08
- changes in Flux backend.
- changes in config json (upscaling confs. tiles sizes).
- backported reeal 'Align Your Steps' scheduler from ComfyUI.
- added additional code to ControlNet to enable it from external scripts.
- reworked Dynamic scheduler extension into Dynamic CFG extension script all-in-one with possibility to setup it from external scripts.
- added additional code into FreeU integrated, now it's FreeU, with possibility to setup it from external scripts.
- reconfigured sampler/scheduler confs for UI presets for txt2img and img2img.
- SD Upscale script moved/changed to ClarityHD Upscale script:
* backported and enhanced Clarity upscaler/enhancer workflow process.
* reworked configs.
* added presets with 3 square tile sizes 768/1024/1152.
* integrated ClarityHD presets into main code, controlnet, freeu, Dynamic CFG, PAGI.
* quick setup: go to img2img -> upscale 1x -> select JaggernautReborn SD1.5 model + SD VAE -> Select ClarityHD script and choose preset - Script will enable and set all values.
- some misc changes in code.

#2025/12/20
- fixed a batch processing bug (this bug was in the original forge, by the way) when the engine reserves more memory than necessary during batch processing, and with each subsequent processing of a new file (image) from a folder/list, memory consumption only increases and it begins to fill up with garbage and the remains of tensor layers and previous layers from past files, and this leads to OOM.

#2025/12/09
- reworked LoRa patching
- implemented LoRa computation cast selection (via drop-down selection in UI and Memory section in Settings):
* HW-based selection: based on hw-detection of your GPU it will be automatically selected between bfloat16 or float32.
* Autocast: automatically selects between bfloat16/float32 based on CUDA availability.
* forced float16/bfloat16/float32 computation.
* NOTE: 
* bfloat16/float16 use less memory and useful on LowVRAM GPUs, but cost of slow LoRa slow first run (next generations with same LoRa will be run with normal generation speed as without LoRa). Useful for use with FLUX and lot of LoRas.
* float32 is faster but cost of high VRAM and RAM usage.
- reworked Main Settings/Memory section: added overview for Memory Cleanup strategy and LoRa computation cast, with appropriative settings.

#2025/10/15
- reworked VAE/CLIPs download menu:
* re-uploaded more clips to Hugginface cloud with more proper names
* addded Illustrius l/g clips
* added new Flux.1 Krea clip_l
* added several detailed clip_l 
- fixed VAE/textEncoders dropdown menu width dependance to models name
- fixed Detail Daemon extension IndexError: out of bounds for axis for self.schedule

#2025/09/30
- fall back to 'Queue' value in Async memory setup by default for all UI profiles.
- added pre-clean memory cache and scum before model load to prevent additional ovefrlow memory.

#2025/09/29
- reworked the structure of startup bat files: now they divided onto three types: 
* main_menu.bat - Main Menu file with models installation. Will be updated from time to time.
* run_normal.bat/run_rtx50.bat - run files, excluded from update by future commits.
* env_normal.bat/env_rtx50.bat - enviroment setup for normal run and run on RTX50xx GPU cards.
* env_normal.bat/env_rtx50.bat will not updated ever by future commits. Here you can setup external path for models, as you're done before in webui-user.bat.
- added SD WebUI Dynamic Prompt Extension into pre-integrated internal extensions (https://github.com/KoinnAI/dynamic-prompting-simplified).
- added auto-noise-schedule extension into pre-integrated internal extensions (https://github.com/michP247/auto-noise-schedule).
- added SADA Extension for Stable Diffusion WebUI Forge into pre-integrated internal extensions (https://github.com/LeDXIII/SADA-Extension-for-Forge-WebUI).
- added ReActorZero extension fork pre-installation while first startup (and re-installing if not present in build).
- fixes in Dockerfile for torch271cu128 and torch280cu128 branches.

#2025/09/26
- fixed flux generation issues due to RMSNorm wrong code for all branches
- disabled pre-congigured upscale model for txt2img mode

#2025/09/21
- fixed downloading models from main .bat menu

#2025/09/18
- fixed requirements re-installation while first install for torch280cu128 branch
- fixed flux generation issues due to RMSNorm wrong code on torch280cu128 branch

#2025/09/11
- reverted back part of code: sampling, K-diffusion, LoRa patcher, upscaler things due to some issues.
- replaced samplers&schedulers for img2img in SD/SDXL UI profiles (onto stable).
- moved Triton from obligate install to optional (with flag --triton in webui-user.bat. use on your own risk!) due to fault to launch engine issue due to xfromers on RTX30x-40x series GPUs. Now work fine with xformers 'out of box'.

#2025/09/04
- reworked part of code.
- moved uv from prerequirements directly into launchutils code.
- setup now works in hybrid mode both uv and pip for faster proces.
- deleted many args from COMMANDLINE_ARGS.
- deleted npu support.
- deleted ngrok support.
- deleted python check, cuda test, and list of rest checups while startup.
- added support to pth and sft models.
- updated forge2 flux kontext extension.
- added MaHiRo CFG Rescaler. Available to hide from Settings/StableDiffusion.
- added CFG Rescaler. Available to hide from Settings/StableDiffusion.
- added and enabled fp16_accumulation (Torch 2.7.1 Cuda 12.8) support (already in COMMANDLINE_ARGS since new version).
- fixed convert RGBA → RGB to JPEG while img2img upscale.
- added Color Correction while img2img upscale.
- moved some int64 into float32 processing.
- deleted some unuseful messages in console, added useful.
- rewoked LoRa patcher backend. Now working with congigurable chunk size (for now via code edit for devs).
- forse some processing from float16 into float32, this helps speed-up low-end GPUs (like my 1660Ti Mobile) with GGUF work and Flux.1 Tools/Kontext.
- moved 'pin_shared_memory' checkbox choice to hidden duw to unuseful. Value is 'CPU' for all UI profiles always.
- reworked most of pre-configured configs.
- raised tile size for Upscalers in Extra tab up to 512pix with ovcerlap 128pix.
- enabled color correction in configs.
- rearranged integrated extensions folders.
- added webUI_ExtraSchedulers extension: https://github.com/DenOfEquity/webUI_ExtraSchedulers by DenOfEquity.
- optimized some code.

#2025/08/30
- webui-user.bat encoding fix (CRLF).
- added Insightface mhl local installation without MSVC compiling need into CU128 branch.
- added packaging and setuptools local mhl files for preinstalling prerequirements both branches.

#2025/08/25
- further improvement of UI profiles logic
- implemented Queue/Async safe logic for GGUF, Kontext, Fill, Depth, Canny checkpoints and LoRas: Async force blocked and changed to Queue if user select any of these model type (parsed by extension and by name, LoRas parsed by prompt). This saves engine from OOM.
- added force fallback to CPU while OOM when changing models to prevent engine OOM.

#2025/08/23
- reworked requirements content. Added new modules.
- reworked launch logic. Added prerequiremets file. 
Moved some modules from pre-installed into prerequiremets for installing before overall installation start.
- due to implmenting prerequiremets fixed issue "no module" for 'packages'.
- fixed issue with 'numpy' module installing in CU128 branch.
- now ZIPs have lower weight due to implementing prerequiremets.
- moved 'uv' module into prerequiremets.
- added 'triton-windows' for install into CU121 branch.
- due to optimizing CU128 became same speed as CU121 branch.
- fixed cmall dropdown for 'checkpoints'.

#2025/08/20
- reworked and fixed 'Memory Cleanup Strategy' dropdown menu list. Now available 7 (seven) choices:
* Smart Purge - same as Smart. but always purges GPU cache before main logic.
* Smart - old 'Smart' method. See changelog for description.
* Always - always clear all possible things from RAM+VRAM (it's old Clear Always method).
* Full - always purges GPU cache and then clear all possible things from RAM+VRAM
* Soft (Cache) - clears RAM cache.
* GPU Cache - clears only GPU cache.
* None (Upscale) - do not clear anything. Best for faster upscale (img2img).
- Smart Purge is best for generating; None (Upscale) best for upscale, expecially if you're using same model for batch upscaling.
- 'Memory Cleanup Strategy' now saves value in config.json, so it saves previous state after UI reboot and even re-start engine.

#2025/08/18
- updated Dockerfile both branches to fit new flags while startup
- added new UI profiles for SD subsection and reworked samplers and schedulers in SD and SDXL subsections
- deleted sd/xl/flux defaults setup in settings to make new option of UI Profiles change everything and save it until next UI startup
- added 'Restart Forge' button (with RED CIRCLE icon) into main UI:
*unloads all modules (stream, processing, memory_management, dynamic_args, Context)
*recreates root_block, shared.state, and all related components
*recalls forge_main_entry() with full reinitialization
*doesn't exit Gradio, refreshing the UI page after 35 seconds (time for full restart)
- added Ultimate SD Upscale script, pre-configured to best upscale
- additionally fixed UI broke when NO CHECKPOINTS FOUND

#2025/08/15
- reworked optimizaton flags in webui-user.bat file: more optimized, faster, more stable, less VRAM usage.
- added 'Clear VRAM/RAM' button into UI: clears memory from cache/etc. just like in ComfyUI. Placed near Checkpoints refresh button.
- added 'Memory Cleanup Strategy' dropdown menu to change free_memory algorithm on-the-fly. With this I've moved .bat menu entry 10. into UI:
- choice of Smart / Clear Always / Native Forge. 'Smart' is by default:
* no memory ovwerflow like in official ForgeSD.
* if the model is not used anywhere else (only in the list and locally), it will be unloaded.
* if the model changes, all memory will be cleaned.
* if the model is not in the keep_loaded list and is on the desired device, it will be unloaded.
* if at least one model was unloaded, the cache will be cleared.
* if the model was not unloaded from memory and if it did not change, it will saved in memory.
* cache will be cleaned at each generation.
* deleted alleavy information messages (about timeings load etc.) this help to make img2img upscale speed faster up to >22%.
- 'Clear Always' - always clears every memnory possible. No memory ovwerflow like in official ForgeSD.
- 'Native Forge' - works same way as native ForgeSD free_memory works.
- replaced UI Profile from Radio-button choice to Dropdown menu. Added more profiles. Reworked all of them, except native ForgeSD with all settings and picture resolutions):
* UI Profiles now save previous settings (except steps and dimensions) next FastForge run/start.
* UI profiles now save GPU Weight, Swap method and swap location values next FastForge run/start.
* -2GB or -3Gb  profile reserves 2Gb or 2.7Gb for inference memory, Dynamic UI profile reserves inference memory dynamically, depending to VRAM load.
- implemented and added Dynamic GPU Weight self-setup for UI Profiles, depending used VRAM.
- added CPU fallback on high VRAM when OOM tp save work.
- dynamic type LoRa load method (also for future LoRa usage patch).
- added modular unloading while changing checkpoint.
- replaced final numpy reinstall after requirements setup, also added protobuf setup at engine first start. Also, implemented sudden update of these modules to re-install them automatically.
- optimized code for py312-torch271-cu128 branch.
- fixed 'xformers' flag in webui-user.bat for py312-torch271-cu128 branch.
- added 'uv' python module to py310-torch231-cu121 branch.
- now both branches will be shared at one Tag and version number.

#2025/08/11
- made new branch - py312-torch271-cu128 - FastForge2, based on latest sources.
- updated python up tp v3.12.7.
- updated rewuirements-versions.
- updated Torch up to v2.7.1 CUDA 12.8 with all python modules.
- updated xformers.
- added triton-windows.
- uptimized release zip size.
- ported diffusers safety gen checkup patch (off) in release version.
- deleted transformers warn in release version.
- fixed models load logic to fit Torch >=v2.6 to prevent embeddings and textural inversions models been corrupted.
- reworked some escape sequences re.compile.
- small fixes.

#2025/08/10
- updated adetailer up to v25.3.0.
- added back some config files to gitignore.
- moved all pre-installed and pre-configured extensions to extensions-builtin folder to prevent uninstalling/modyfing/updating by user and for making user-extensions space mor understandable.
- added more explanations in main .bat menu for some entries
- updated requirements-versions

#2025/08/09

STRONGLY RECOMMENDED TO CLEAR RE-INSTALL DUE TO POSSIBLE ISSUES AFTER UPDATE!

- changes in main requirements:
* replaced controlnet-aux with own patched version to fit fresh timm module for support Florence-2 extension.
- deleted all Forge SD Space internal extensions EXCEPT Florence-2. Work even on 6Gb GPU (tested on GTX1660Ti Mobile 6Gb GDDR6).
- deleted Info some message in Detail Demon extension.
- deleted Info some messages in TeaCache extension.
- deleted Cleaner extension (for Release build).
- moved Facebook offline translate extension to download and install option via main .bat menu (it will be downloaded and unpacked with own model into extension folder).
- deleted pre-loaded 3D Pose extension files. They will be downloaded at first start, or if they will be lost.
- created and implemented new Memory Clear profile - 'Smart' for free_memory while work. Now memory will be cleaned:
* if the model is not used anywhere else (only in the list and locally), it will be unloaded.
* if the model changes, all memory will be cleaned.
* if the model is not in the keep_loaded list and is on the desired device, it will be unloaded.
* if at least one model was unloaded, the cache will be cleared.
* if the model was not unloaded from memory and if it did not change, it will saved in memory.
* cache will be cleaned at each generation.
* deleted alleavy information messages (about timeings load etc.) this help to make img2img upscale speed faster up to >22%
- new Memory Clear profile - 'Smart' now is work by default, 'out-of-box', also can be changed via main .bat menu. 
- deleted duplicate enviroment in webui-user.bat
- changed Main Profile to SD by default (instead of Flux) for proper memory management by default.
- changed Memory Optimization profile to ALL VRAM -2Gb Async/CPU for proper memory management by default.
- small changes and fixes in webui-user file (main .bat menu).
- enabled back models hashiing for better experience.
- added check and update PIP, if necessary, on every FastForge start.
- release file size optimizations.

#2025/07/28
- updated Flux.1 Kontext extension script

#2025/07/12
- Synced manually with Forge as of July 12, 2025 — 10 upstream commits incorporated via empty commits to preserve local customizations.

#2025/07/10
- added new Chroma config.json files
- added Download/Install Chroma VAE entry into BAT menu (MENU/Flux.1 VAE/CLIPS)
- fixed Flux.1 VAE/CLIPS submenu entries
- small changes in configs for inpaint/etc.

#2025/07/03
- added separate cuda stream for live preview VAE (ported commit)
- added line to update sampling ste (ported commit)
- added support wd_on_output for DoRA (ported commit)
- added BF16 to GGUF (ported commit)
- resolve warnings of datetime library (ported commit)
- repair ancestral sampling for FLUX (ported commit)
- added Chroma (ported commit)
- added multithreading softinpainting (ported commit)
- added support for fp8 scaled (ported commit)
- fixed SD upscale Batch count (ported commit)
- added Block Cache and TeaCache extension
- added Flux Kontext extension
- set by default Clear ALways for RAM Management
- set by default VRAM-2Gb Async+CPU profile
- added Download/Install Flux.1 Kontext fp8/fp16/GGUF entries into BAT-manu (for portable version)
- re-worked Flux.1 Text Encode clips download menu, added fp8 CLIP, also pointed fp16 clip

#2025/05/08
- inpaint resolution (for Flux mainly) reduced to 768x768, it will brings generation speedup x2.5 without Flux inpaint quality loss by my tests
- changed resolution for SDXL by default

#2025/04/19
- some lexic fixes in main menu (webui-user.bat)
- deleted Cleaner extension from sources, available now only in portable build)

#2025/04/16
- added some repositories back to sources for install while first start
- deleted non-needed code from webui-user.bat about four mandatory repos check/install
- added Docker file for Docker run support
- some changes in requirements_versions
- reconfigured gitignore to prevent updating .bat files, memory_management and main_entry files

#2025/04/14
- enabled always show GPU Weights slider for SD/XL profiles (lowering weights helps while upscaling to great resolutions on LowRAM GPU)
- enabled ClipSkip slider for XL profile
- enlarged expand dimensions in Mosaic Outpaint extension
- fixes with webui-user.bat encoding (that crashes menu)

#2025/04/13
- enlarged FluxTools Fill outpaint expand max size to 2048px all sides (top/bottom/left/right)
- added some pre-confs into ui-config.json

#2025/04/11
* release for own Git repo
* moved main menu and some files from START.bat to webui folder and webui-user.bat for availability for Git users
* reworked python folder and requirements_versions, deleted some not not needed python modules
* fixed FluxTools Redux work
* ultralytics python module moved from pre-installed to requiremets
* reworked START.bat menu:
- deleted FlusTools Redux model part due to user can use now usual Flux.1 Dev fp8/fp16/GGUF and Flux Schnell
- replaced FluxTools Fill, Canny, Depth "open in browser" links to HigginFace repos with GGUFs for these models
- some code fixes for downloaded models
* some changes for webui-user.bat file
* Memory Management profile now set to "Always Clear Memory" by default
* deleted BiRefNet due to incompatible with new requiremets (dev updated to pytorch 2.5*)

#2025/04/09
* additional changes in main settings, UI conf, webui-user.bat files
* added additional keys with explanations into webui-user.bat file
* enlarged pictures size limits up to 4096x4096 for almost all
* re-configured 'sd-webui-prompt-format' extension: disabled 'remove underscores' by default
* deleted FluxTools module
* deleted FreeU module
* Integrated modules: 
- Aspect Ratio
- FluxTools v2, workable on LOW VRAM (tested with GTX1660Ti Mobile 6Gb + 64Gb RAM)
- additional samplers
- seamless inpainting
- replacement for FreeU
* small changes in START.bat menu:
- deleted text_encoders_FP8.zip download option, due to no need now
- added option (step (9)) to change MEMORY MANAGENT config: Native ForgeSD/Always Clear Memory (this helps to prevent VRAM/RAM overflow)
- fixes in menu code
* reworked RAM optimizations profiles:
- now there are normal ForgeSD profile, optimized with changed picture dimensions, and TOTAL_VRAM-1GB/-2GB 
- TOTAL_VRAM-3Gb profile has been deleted
- also, raised WEIGHTS for SD profile (SD/SDXL/FLUX/ALL) in all optimized profiles

#SOME EXPLANATIONS:
- if you're used previous version, just move old models into new, except 'models\diffusers' folder.
New FluxTools will download needed filess when you will use Fill, Redux, Canny first time (it's about 3.6Gb).
Main Cann, Redux, Fill, Depth models are same as before.

- if you have 6-8Gb VRAM and 32-128Gb RAM you can also use new FluxTools:
choose needed model (for example Flux Fill for outpaint with Fill), mode WEIGHTS SLIDER to 1Gb, and change memory profile to ASYNC+CPU.
Generation will take some time, but it will work for you. Tested on my laptop with GTX1660Ti Mobile 6Gb + 64Gb RAM.

- If you had some VRAM and RAM overflow problems, go to step 9 in main START.bat menu and choose ALWAYS CLEAR MEMORY.
This may help with overflow.

#2025/04/03
* optimized rewuirements list
* deleted and updated some python modules
* reconfigured some settings by default in config file
* added additional menu in to bat file: download Flux VAE, Flux CLIP I, CLIP II (FP16-based, universal), and CLIP I Detailed for more detailis.
Links provided by my HugginFace cloud folder.
* fixed unpack error for archive

#2025/04/02
* torch moved to requirements instead of preconfigured in build (Forge will download Torch while first start and configure it to fit your PC)
* moved onnxruntime to real GPU version, also moved to requirements instead of pre-congigured in build
* moved bitsandbytes to requirements instead of pre-congigured in build
* updated PIP

#2025/03/30
* added additional upscalers pack download option into main menu/upscalers download
OLD VERSION USERS CAN ADD NEW MENU BY REPLACING AN OLD START.bat with NEW ONE!
* added empty Lora folder by default
* reworked RAM optimization profiles (main menu, step (7)):
- moved to Queue+CPU and Async+CPU profiles as most stable and without RAM/VRAM leak
OLD VERSION USERS CAN REPLACE PROFILES BY DELETING OLD AND COPYING WEBUI\ram_opt FOLDER!

#2025/03/29
* reworked python requirements, and inbuild modules, optimized code
* fixed RAM/VRAM leakage in memory optmized profiles (in MAIN MENU, step (7)). Now working fine with SD/XL/IL/FLUX.
* raised speed generation for SD/SDXL/IL/FLUX, especially for Flux fp8 and greatly for Flux fp16 models (from 15 minutes up to to 6-7 minutes on GTX 1660Ti Mobile 6Gb)
* disabled Auto-update changes option by default in Prompt Translate module
* added DenOfEquity's HyperTile extension, adapted to last Forge sources (https://github.com/DenOfEquity/HyperTile)

#2025/03/26
* moved from curl onto aria2c downloader, thx to @NeuroDonu for info
* reworked start script, extended menu
* added option download nf4v2 model by script (aria2c) or open in Browser
* added AcademiaSD' Flux ControlNet extension modded by @li_aeron (me). WARN! Work with 8Gb VRAM and up!
- You have to install all needed models, also download text_encoders_FP8.zip via MAIN MENU. 
Optionally, you can create HugginFace access token and place it into file webui\huggingface_access_token.txt, but it's not necessary in my modded FLux Tools extension. 
* reworked requirements_versions to fit new build with Flux ControlNet extension
* in MAIN MENU there was added new menu to download models and text_encoders_FP8 zip for Flux ControlNet, with choice to download by own or open in Browser
* some modules code optimization
* added Physton prompt translator extension (https://github.com/Physton/sd-webui-prompt-all-in-one prompt) with option to download pretrained offline (facebook) model for it in MAIN MENU

#2025/03/25
* reworked start script, now it has own menu:
*** ability to download two upscale models downloaded at first start from hugginface/LeeAeron (downloaded into models/DAT)
*** ability to download 8 additional adetailer models downloaded at first start from hugginface/LeeAeron (downloaded into models/adetailer)
*** ability to download Flux.1 Dev NF4v2 model with inbuilt VAE and CLIP models (11gb) (downloaded into models/Stable-diffusion)
*** ability to reconfiigure RAM optimizations profiles: pure ForgeSD/optimized (normal) ForgeSD/for PC with high RAM low VRAM (Async+Shared/Queue+Shared)
* reworked system folder, now very close to official Forge SD, this helps to make unpacked build weight less up to 0.5Gb
* now build will download and install inside itself all needed python modules, including xformers
* cleared some files as no needed
* deleted some warns and notif messages in Gradio and Detail Daemon
* added some performance optimizations into web-user.bat file
* pure Forge SD RAM/image dimensions by default

#2025/03/22
* added Digiearts sd-forge-cleaner extension
* added DenOfEquity superPrompter-webUI extension
* added Haoming02 sd-webui-prompt-format extension

#2025/03/09
* updated Git sources 2025/03/09
* reworked some Python modules and files
* fixed ControlNet work for SD/SDXL/IL models (OpenPose etc.)
* deleted some unneeded files

#2025/03/01
* updated Git sources 2025/02/28
* deleted RealESRGAN_x4plus, ScuNET, SwinIR_4x as useless
* deleted duplicated embeddings for sd/xl (negative etc)
* deleted some empty folders
* fixed auto-setup VRAM lvl for profiles by default

#2025/02/27
* forked from official Forge Git stable build with CUDA 12.1 and Pytorch 2.3.1
* included all last updates from Git by 2025/02/27
* added and enabled working xformers
* enabled Cuda Stream, Cuda Malloc for engine
* added Adetailer models (all that I found in inretnet) and replaced cached models to files itself, this make faster UI launch
* added ADetailer 'Eyes' model
* deleted nsfw / watermark checker code in diffusers Python module (in Portable Version)
* added img2img HiRes Fix. fully working version
* added TeaCache
* added 3D openPose Editor
* added Detail Demon. Own pre-configured version
* added Mosaic Outpaint
* added SD Upscale
* added 4xFFHQDAT, 4xSSDIRDAT, RealESRGAN_x4plus, ScuNET, SwinIR_4x upscale models
* pre-configured configs and settings
* added some embeddings
* unlocked deleted 'Flux Realistic' samplers with pre-integrated Google code for Flux.1 / Flux.S models
* re-linked all chanes to intermal webui_cache folder for clearer and better engine work
