// Two-Pass Shock Propagation (2PSP) minimal demo for Fig.1,2,3
// Libraries: freeglut (GLUT), GLM, Eigen only.
// Camera, mouse/keyboard usage are implemented in a style similar to the provided gheat.cpp.  (see usage())
// Reference implementation notes in comments map to the paper's equations/pseudocode:
//   - Alg.2 SolveContactShock (up & down passes)  [paper Alg.2]  :contentReference[oaicite:6]{index=6}
//   - Alg.3 SolveContactTopBody (per-layer fixed point on top body)  [paper Alg.3]  :contentReference[oaicite:7]{index=7}
//   - Eq. (1) generalized inverse mass F_d                              :contentReference[oaicite:8]{index=8}
//   - Eq. (3) constraint value C_d (here named g_d)                     :contentReference[oaicite:9]{index=9}
/* - Eq. (4) GS update (only for reference; 2PSP uses Eq.(7) in TopBody solve)  :contentReference[oaicite:10]{index=10}
   - Eq. (6) friction cone projection (tangent projection)               :contentReference[oaicite:11]{index=11}
   - Eq. (7) 2PSP update with bottom infinite mass (F_bottom=0)         :contentReference[oaicite:12]{index=12}
*/
// Camera/UI style inspired by the uploaded gheat.cpp viewer. :contentReference[oaicite:13]{index=13}

#if defined(WIN32)
  #pragma warning(disable:4996)
  #include <GL/freeglut.h>
#elif defined(__APPLE__) || defined(MACOSX)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #define GL_SILENCE_DEPRECATION
  #include <GLUT/glut.h>
#else
  #include <GL/freeglut.h>
#endif

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ====== Types / Math ======
using V3 = glm::vec3;
using Quat = glm::quat;

static inline float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }

// ====== Rigid Body / Contact ======
struct RigidBody {
  V3 x;           // position
  Quat q;         // orientation
  V3 v;           // linear velocity
  V3 w;           // angular velocity
  float m;        // mass
  float invM;     // inverse mass (0 for ground/infinite)
  Eigen::Matrix3f Iw;     // world inertia
  Eigen::Matrix3f IwInv;  // inverse
  V3 half;        // half extents (draw)
  bool isGround = false;
};

struct Contact {
  int idxBot, idxTop;     // bottom/top body indices (as in the paper)
  V3 pBot, pTop;          // contact points (world). In this demo we predefine.
  V3 n, t1, t2;           // normal up, two tangents
  float phi;              // penetration depth (>0 means penetration)
  // Lagrange multipliers for normal & tangents (accumulated inside layer solve)
  float lambdaN=0.f, lambdaT1=0.f, lambdaT2=0.f;
  // per-iteration delta (temporary)
  float dN=0.f, dT1=0.f, dT2=0.f;
  // Unused impulses accumulated for bottom (Upward pass) to be applied in Downward pass
  V3 Jbot_accum = V3(0);
};

// ====== Scene: three boxes and ground, fixed contacts like Fig.1?3 ======
static std::vector<RigidBody> gBodies;
static std::vector<Contact>   gContacts;
static std::vector<std::vector<int>> gLayers; // layer0: ground-X, layer1: X-Y, layer2: Y-Z

// ====== Globals / Params ======
static const float h  = 1.0f/60.0f; // timestep (for Eq.(3))
static const float mu = 0.5f;       // friction coefficient (for Eq.(6))
static const float epsConverge = 1e-6f;
static const int   maxIterPerLayer = 75;

// ====== Camera  ======
static int gWinW = 1280, gWinH = 800;
static float camDist = 0.6f;
static float yawRad = 0.6f, pitchRad = 0.3f;
static float panX = 0.f, panY = 0.f;
static bool lbtn=false, rbtn=false;
static int lastX=0, lastY=0;
static glm::mat4 gProj(1), gMV(1), gMVP(1);

static void usage(const char* argv0){
  std::puts("=== Two-Pass Shock Propagation for Stable Stacking with Gauss-Seidel (Fig.1-3) Minimal Demo ===");
  std::printf("Usage: %s\n", argv0);
  std::puts("\nMouse:");
  std::puts("  Left-drag   : rotate camera (yaw/pitch)");
  std::puts("  Right-drag  : pan");
#if defined(FREEGLUT)
  std::puts("  Wheel       : zoom");
#endif
  std::puts("\nKeyboard:");
  std::puts("  1 : Run Upward pass only (Fig.1) and redraw");
  std::puts("  2 : Run Downward pass (Fig.2) on static case and redraw");
  std::puts("  3 : Run Downward pass (Fig.3) dynamic case (slight imbalance) and redraw");
  std::puts("  r : Reset scene to initial stacked configuration");
  std::puts("  + / - : zoom in / out");
  std::puts("  q / ESC : quit");
  std::puts("");
}

// ====== Utilities ======
static Eigen::Matrix3f skew(const Eigen::Vector3f& a){
  Eigen::Matrix3f S;
  S <<    0, -a.z(),  a.y(),
        a.z(),    0, -a.x(),
       -a.y(), a.x(),    0;
  return S;
}

// Compute world inertia of an axis-aligned box (about COM) given orientation q and body-space inertia.
// For simplicity we use a uniform box inertia diag in body, then rotate to world.
static void updateWorldInertia(RigidBody& b){
  // Box inertia in body space: I_body = (1/12)*m*diag( (y^2+z^2), (z^2+x^2), (x^2+y^2) )
  float x2 = (2*b.half.x)*(2*b.half.x);
  float y2 = (2*b.half.y)*(2*b.half.y);
  float z2 = (2*b.half.z)*(2*b.half.z);
  float c = (b.m>0.f)? (b.m/12.f) : 1.f;
  Eigen::Matrix3f Ib;
  Ib.setZero();
  Ib(0,0) = c*(y2+z2);
  Ib(1,1) = c*(z2+x2);
  Ib(2,2) = c*(x2+y2);

  // R (glm) -> Eigen
  glm::mat3 Rg = glm::mat3_cast(b.q);
  Eigen::Matrix3f R; 
  for(int i=0;i<3;i++)for(int j=0;j<3;j++) R(i,j) = Rg[j][i]; // note: glm is column-major
  b.Iw = R * Ib * R.transpose();

  if(b.isGround || b.invM==0.f){ // infinite mass => zero inverse inertia
    b.IwInv.setZero();
  }else{
    b.IwInv = b.Iw.inverse();
  }
}

// Apply linear/angular impulse J at contact point r (from COM) to body velocities (v,w).
static void applyImpulse(RigidBody& b, const Eigen::Vector3f& r, const Eigen::Vector3f& J){
  if(b.invM==0.f) return; // ground / infinite
  Eigen::Vector3f dv = b.invM * J;
  Eigen::Vector3f dw = b.IwInv * (r.cross(J));
  b.v += V3(dv.x(), dv.y(), dv.z());
  b.w += V3(dw.x(), dw.y(), dw.z());
}

// ====== Constraint evaluation ======
// Generalized inverse mass F_d (Eq.(1)) for one body given contact arm r and direction d.  :contentReference[oaicite:14]{index=14}
static float generalizedInvMass(const RigidBody& b, const Eigen::Vector3f& r, const Eigen::Vector3f& d){
  if(b.invM==0.f) return 0.f;
  Eigen::Vector3f c = r.cross(d);
  // F_d = 1/m + (r x d)^T * IwInv * (r x d)
  Eigen::Vector3f t = b.IwInv * c;
  return b.invM + c.dot(t);
}

// Evaluate constraint value g_d (Eq.(3)) for dir d in {n, t1, t2}.   :contentReference[oaicite:15]{index=15}
// 3 + n^T(r_top - r_bot) = h * n^T( v_top - v_bot )  => rearranged to velocity-level residual g_d
static float evalConstraint(const RigidBody& bot, const RigidBody& top,
                            const Eigen::Vector3f& rBot, const Eigen::Vector3f& rTop,
                            const Eigen::Vector3f& d, float phi){
  // point velocities
  Eigen::Vector3f vBot = Eigen::Vector3f(bot.v.x, bot.v.y, bot.v.z) + bot.IwInv*Eigen::Vector3f(0,0,0); // angular term handled below
  Eigen::Vector3f vTop = Eigen::Vector3f(top.v.x, top.v.y, top.v.z);
  // add angular contributions É÷ Å~ r
  vBot += Eigen::Vector3f(bot.w.x, bot.w.y, bot.w.z).cross(rBot);
  vTop += Eigen::Vector3f(top.w.x, top.w.y, top.w.z).cross(rTop);

  float rel = d.dot( vTop - vBot ); // d^T (r'_top - r'_bot)
  // g_d = d^T(vTop - vBot) - (phi + d^T(xTop - xBot))/h
  // In this minimal demo, we treat (d^T(xTop - xBot))?0 around contact and use penetration phi only.
  float g = rel - (phi)/h;
  return g;
}

// Project friction (Eq.(6)) onto cone after updates to lambdas.  :contentReference[oaicite:16]{index=16}
static void projectFriction(float mu, float lambdaN, float& lambdaT1, float& lambdaT2){
  float lim = mu * std::max(0.f, lambdaN);
  float mag = std::sqrt(lambdaT1*lambdaT1 + lambdaT2*lambdaT2);
  if(mag > lim && mag>1e-12f){
    float s = lim / mag;
    lambdaT1 *= s;
    lambdaT2 *= s;
  }
}

// ====== Alg.3 : SolveContactTopBody(layer) ======
// In 2PSP, bottom is treated as infinite mass in BOTH passes; hence É¢ = -g_d / F_top,d  (Eq.(7)).  :contentReference[oaicite:17]{index=17}
static void SolveContactTopBody(const std::vector<int>& layer){
  for(int it=0; it<maxIterPerLayer; ++it){
    float maxDelta = 0.f;

    // Normal first (Poulsen10 guideline, also in Alg.1) then tangents; we do per-contact loop twice.
    for(int pass=0; pass<2; ++pass){
      for(int ci : layer){
        Contact& c = gContacts[ci];
        RigidBody& bot = gBodies[c.idxBot];
        RigidBody& top = gBodies[c.idxTop];

        // arms from COM(Center Of Mass) to contact points
        Eigen::Vector3f rBot = Eigen::Vector3f(c.pBot.x - bot.x.x, c.pBot.y - bot.x.y, c.pBot.z - bot.x.z);
        Eigen::Vector3f rTop = Eigen::Vector3f(c.pTop.x - top.x.x, c.pTop.y - top.x.y, c.pTop.z - top.x.z);

        // directions
        Eigen::Vector3f n (c.n.x,  c.n.y,  c.n.z );
        Eigen::Vector3f t1(c.t1.x, c.t1.y, c.t1.z);
        Eigen::Vector3f t2(c.t2.x, c.t2.y, c.t2.z);

        // generalized inverse mass of TOP only (bottom is infinite => F_bottom=0), Eq.(7)
        float Fn_top  = generalizedInvMass(top, rTop, n );
        float Ft1_top = generalizedInvMass(top, rTop, t1);
        float Ft2_top = generalizedInvMass(top, rTop, t2);
        if(Fn_top < 1e-12f) Fn_top = 1e-12f;
        if(Ft1_top< 1e-12f) Ft1_top= 1e-12f;
        if(Ft2_top< 1e-12f) Ft2_top= 1e-12f;

        if(pass==0){
          // --- Normal solve (Eq.(7) using Eq.(3) residual) ---
          float gN = evalConstraint(bot, top, rBot, rTop, n, c.phi);     // Eq.(3) residual along n
          float dN = - gN / Fn_top;                                      // Eq.(7) update on TOP
          // Non-penetration clamp like GS (ensure lambdaN >= 0)
          float newLambdaN = std::max(0.f, c.lambdaN + dN);
          dN = newLambdaN - c.lambdaN;
          c.lambdaN = newLambdaN;

          // Apply impulse only to TOP; accumulate equal-opposite for BOTTOM (Upward semantics; Alg.3 lines 4?6)
          Eigen::Vector3f J = dN * n;
          applyImpulse(top, rTop, J);
          c.Jbot_accum += V3(-J.x(), -J.y(), -J.z()); // store unused bottom impulse for downward pass

          maxDelta = std::max(maxDelta, std::fabs(dN));
        }else{
          // --- Tangential solve, with friction cone projection (Eq.(6)) ---
          float gT1 = evalConstraint(bot, top, rBot, rTop, t1, 0.0f);
          float gT2 = evalConstraint(bot, top, rBot, rTop, t2, 0.0f);

          float dT1 = - gT1 / Ft1_top;
          float dT2 = - gT2 / Ft2_top;

          float newT1 = c.lambdaT1 + dT1;
          float newT2 = c.lambdaT2 + dT2;

          // project onto cone with mu and updated normal (Eq.(6))
          projectFriction(mu, c.lambdaN, newT1, newT2);

          dT1 = newT1 - c.lambdaT1;
          dT2 = newT2 - c.lambdaT2;
          c.lambdaT1 = newT1; c.lambdaT2 = newT2;

          Eigen::Vector3f J = dT1 * t1 + dT2 * t2;
          applyImpulse(top, rTop, J);
          c.Jbot_accum += V3(-J.x(), -J.y(), -J.z());

          maxDelta = std::max(maxDelta, std::max(std::fabs(dT1), std::fabs(dT2)));
        }
      }
    }
    if(maxDelta < epsConverge) break; // fixed point within layer (Alg.3 outer loop stop)  :contentReference[oaicite:18]{index=18}
  }
}

// ====== Alg.2 : Upward then Downward (per-layer) ======  :contentReference[oaicite:19]{index=19}
static void SolveContactShock_UpwardOnly(){
  // Upward: bottom->top, layer by layer; call SolveContactTopBody(layer)
  for(const auto& layer : gLayers){
    SolveContactTopBody(layer);
  }
}

static void SolveContactShock_DownwardOnly(){
  // Downward: top->bottom; first SolveContactTopBody(layer), then apply accumulated bottom impulses and reset.
  for(int L=(int)gLayers.size()-1; L>=0; --L){
    const auto& layer = gLayers[L];
    SolveContactTopBody(layer); // top fixed point for this layer

    // Apply accumulated bottom impulses (Alg.2 lines 8-11)
    for(int ci : layer){
      Contact& c = gContacts[ci];
      if(glm::dot(c.Jbot_accum, c.Jbot_accum) > 0.f){
        RigidBody& bot = gBodies[c.idxBot];
        Eigen::Vector3f rBot = Eigen::Vector3f(c.pBot.x - bot.x.x, c.pBot.y - bot.x.y, c.pBot.z - bot.x.z);
        Eigen::Vector3f J (c.Jbot_accum.x, c.Jbot_accum.y, c.Jbot_accum.z);
        applyImpulse(bot, rBot, J);
        c.Jbot_accum = V3(0);
      }
    }
  }
}

// ====== Scene Setup ======
static void resetScene(bool dynamicTilt=false){
  gBodies.clear();
  gContacts.clear();
  gLayers.clear();

  // Ground (infinite)
  RigidBody g;
  g.x = V3(0,0,0); g.q = Quat(1,0,0,0);
  g.v = V3(0); g.w = V3(0);
  g.m = 0.f; g.invM = 0.f; g.isGround = true;
  g.half = V3(5,0.5f,5);
  g.Iw.setZero(); g.IwInv.setZero();
  gBodies.push_back(g);

  auto makeBox = [&](V3 pos){
    RigidBody b;
    b.x = pos; b.q = Quat(1,0,0,0);
    b.v = V3(0); b.w = V3(0);
    b.m = 0.064f; b.invM = 1.0f/b.m;
    b.half = V3(0.02f,0.02f,0.02f); // 4cm box
    updateWorldInertia(b);
    return b;
  };

  // Three boxes stacked: X on ground, Y on X, Z on Y
  RigidBody X = makeBox(V3(0, g.half.y + 0.02f, 0));                        // rests on ground
  RigidBody Y = makeBox(V3(0, g.half.y + 0.02f + 0.04f, 0));               // on X
  RigidBody Z = makeBox(V3(0, g.half.y + 0.02f + 0.08f, 0));               // on Y
  if(dynamicTilt){
    // small lateral offset to mimic dynamic case in Fig.3 (staircase/topple tendency)
    Y.x.x += 0.01f;  // 1cm lateral shift
  }
  updateWorldInertia(X); updateWorldInertia(Y); updateWorldInertia(Z);
  gBodies.push_back(X); // idx 1
  gBodies.push_back(Y); // idx 2
  gBodies.push_back(Z); // idx 3

  // Contacts: (ground-X), (X-Y), (Y-Z)
  auto addContact = [&](int idxBot, int idxTop, float yLevel){
    Contact c;
    c.idxBot = idxBot; c.idxTop = idxTop;
    c.n = V3(0,1,0);
    // orthonormal tangents
    c.t1 = V3(1,0,0);
    c.t2 = V3(0,0,1);
    // contact points directly below/above COM for this minimal demo
    c.pBot = V3(gBodies[idxBot].x.x, yLevel, gBodies[idxBot].x.z);
    c.pTop = V3(gBodies[idxTop].x.x, yLevel, gBodies[idxTop].x.z);
    // penetration depth É” (positive means interpenetration along n)
    // For a tight contact, set a small positive (e.g., 1e-4m) to let the solver push out.
    c.phi = 1e-4f;
    gContacts.push_back(c);
  };

  // y contact levels (top of ground slab at y=0.5, then + 0.04 per box height)
  float y0 = 0.5f;             // ground top
  addContact(0, 1, y0);
  addContact(1, 2, y0 + 0.04f);
  addContact(2, 3, y0 + 0.08f);

  // Layers: layer0: (0-1), layer1: (1-2), layer2: (2-3)
  gLayers.resize(3);
  gLayers[0].push_back(0);
  gLayers[1].push_back(1);
  gLayers[2].push_back(2);
}

// ====== Integration (semi-implicit Euler for visualization) ======
static void integrateAll(){
  for(auto& b : gBodies){
    if(b.invM==0.f) continue;
    // v,w already updated by impulses. Integrate pose:
    b.x += h * b.v;
    // simple angular integration: q <- q + 0.5*dt*(É÷ ? q)
    V3 w = b.w;
    float ang = glm::length(w);
    if(ang > 1e-8f){
      float half = 0.5f * h * ang;
      V3 axis = w / ang;
      Quat dq = glm::angleAxis(2.0f*half, axis);
      b.q = glm::normalize(dq * b.q);
    }
    updateWorldInertia(b);
  }
}

// ====== Rendering ======
static void setProjectionAndView(){
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double aspect = (double)gWinW/(double)gWinH;
  gluPerspective(45.0, aspect, 0.01, 1000.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glm::mat4 R = glm::yawPitchRoll(yawRad, pitchRad, 0.0f);
  glm::vec3 sc(0,0.75f,0);
  glm::vec3 eye = sc + glm::vec3(R * glm::vec4(0, 0.5, camDist, 1));
  glm::mat4 V = glm::lookAt(eye, sc, glm::vec3(0,1,0));
  glm::mat4 T = glm::translate(glm::mat4(1), glm::vec3(panX, panY, 0));
  gMV = V * T;

  glLoadMatrixf(&gMV[0][0]);
}

static void drawBox(const RigidBody& b, const float col[3]){
  glPushMatrix();
  glm::mat4 M(1.0f);
  M = glm::translate(M, b.x);
  M *= glm::mat4_cast(b.q);
  M = glm::scale(M, b.half*2.0f);
  glMultMatrixf(&M[0][0]);
  glColor3fv(col);
  glutSolidCube(1.0);
  glPopMatrix();
}

static void drawGround(const RigidBody& g){
  glPushMatrix();
  glm::mat4 M(1.0f);
  M = glm::translate(M, g.x);                  // center ground at y=0
  M = glm::scale(M, g.half*2.0f);
  glMultMatrixf(&M[0][0]);
  float col[3]={0.25f,0.3f,0.32f};
  glColor3fv(col);
  glutSolidCube(1.0);
  glPopMatrix();
}

static void drawImpulses(){
  glDisable(GL_LIGHTING);
  glLineWidth(2.f);
  glBegin(GL_LINES);
  for(const auto& c : gContacts){
    // visualize stored bottom impulses (red) and top last-iteration direction (yellow approx)
    if(glm::dot(c.Jbot_accum, c.Jbot_accum) > 1e-12f){
      glColor3f(1,0,0);
      V3 a = c.pBot; V3 b = c.pBot + 0.5f * c.Jbot_accum; // scale for visibility
      glVertex3fv(&a.x); glVertex3fv(&b.x);
    }
    // normal direction guide
    glColor3f(1,1,0);
    V3 a2=c.pTop; V3 b2=c.pTop + 0.05f*c.n;
    glVertex3fv(&a2.x); glVertex3fv(&b2.x);
  }
  glEnd();
}

static void display(){
  glViewport(0,0,gWinW,gWinH);
  glClearColor(0.08f,0.08f,0.1f,1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  setProjectionAndView();

  drawGround(gBodies[0]);
  float cx[3]={0.6f,0.7f,0.9f};
  float cy[3]={0.6f,0.9f,0.6f};
  float cz[3]={0.9f,0.6f,0.6f};
  drawBox(gBodies[1], cx);
  drawBox(gBodies[2], cy);
  drawBox(gBodies[3], cz);

  drawImpulses();

  glutSwapBuffers();
}

static void reshape(int w,int h){ gWinW = std::max(1,w); gWinH = std::max(1,h); glutPostRedisplay(); }

static void keyboard(unsigned char key,int,int){
  if(key==27 || key=='q') std::exit(0);
  else if(key=='+'){ camDist *= 0.9f; }
  else if(key=='-'){ camDist *= 1.1f; }
  else if(key=='r'){ resetScene(false); glutPostRedisplay(); return; }
  else if(key=='1'){
    // Fig.1: Upward pass only, static case
    // (Bodies start stacked; we do only upward to see top-only motions & stored bottom impulses)
    SolveContactShock_UpwardOnly(); integrateAll(); glutPostRedisplay();
  }
  else if(key=='2'){
    // Fig.2: Downward pass only, static case
    SolveContactShock_DownwardOnly(); integrateAll(); glutPostRedisplay();
  }
  else if(key=='3'){
    // Fig.3: Dynamic case downward (start with a slight imbalance)
    resetScene(true);
    SolveContactShock_DownwardOnly(); integrateAll(); glutPostRedisplay();
  }
}

static void mouse(int b,int s,int x,int y){
  if(b==GLUT_LEFT_BUTTON)  lbtn = (s==GLUT_DOWN);
  if(b==GLUT_RIGHT_BUTTON) rbtn = (s==GLUT_DOWN);
  lastX=x; lastY=y;
}
static void motion(int x,int y){
  int dx=x-lastX, dy=y-lastY; lastX=x; lastY=y;
  if(lbtn){
    yawRad   += dx * 0.005f;
    pitchRad += dy * 0.005f;
    pitchRad = clampf(pitchRad, -1.55f, 1.55f);
  }
  if(rbtn){
    panX += dx * 0.002f * camDist;
    panY -= dy * 0.002f * camDist;
  }
  glutPostRedisplay();
}
#if defined(FREEGLUT)
static void wheel(int wheel,int dir,int x,int y){
  (void)wheel; (void)x; (void)y;
  camDist *= (dir>0)?0.9f:1.1f;
  camDist = std::max(0.1f, camDist);
  glutPostRedisplay();
}
#endif

int main(int argc,char** argv){
  usage(argv[0]);
  glutInit(&argc, argv);
#if defined(FREEGLUT)
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowSize(gWinW, gWinH);
  glutCreateWindow("Two-Pass Shock Propagation for Stable Stacking with Gauss-Seidel");

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_LINE_SMOOTH);

  resetScene(false);

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
#if defined(FREEGLUT)
  glutMouseWheelFunc(wheel);
#endif

  glutMainLoop();
  return 0;
}
